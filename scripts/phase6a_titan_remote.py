"""Bounded H5 streaming to the authorized GPU server; E drive keeps originals.

TITAN weights are temporarily transferred with explicit user authorization and
deleted from the server after the formal run. Clinical targets stay on server.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import shlex
import threading
import time

import pandas as pd
import paramiko

ROOT = Path(__file__).resolve().parents[1]
SERVER_REPO = "/root/diffusion_phase5_20261007"
REMOTE = "/root/phase6a_titan_runtime"
CLINICAL = "/root/autodl-tmp/diffusion_data/data/data_8/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt"


def connect(args):
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home()/".ssh/known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(args.host, port=args.port, username="root", key_filename=str(args.key),
                   look_for_keys=False, allow_agent=False, timeout=20)
    client.get_transport().set_keepalive(20)
    return client


def mkdirs(sftp, destination):
    current = PurePosixPath("/")
    for part in PurePosixPath(destination).parts[1:]:
        current = current/part
        try:
            sftp.stat(str(current))
        except FileNotFoundError:
            try:
                sftp.mkdir(str(current))
            except OSError:
                # Concurrent upload channels may create the same parent.
                sftp.stat(str(current))


def put(sftp, source, destination):
    mkdirs(sftp, str(PurePosixPath(destination).parent))
    temporary = destination+".partial"
    sftp.put(str(source), temporary)
    sftp.posix_rename(temporary, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="connect.westb.seetacloud.com")
    parser.add_argument("--port", type=int, default=50715)
    parser.add_argument("--key", type=Path, default=Path.home()/".ssh/codex_m2_transfer_ed25519")
    parser.add_argument("--features", type=Path, default=Path(r"E:\diffusion_conch_v15_features"))
    parser.add_argument("--model", type=Path, default=Path(r"E:\diffusion_phase6a_titan\model"))
    parser.add_argument("--cache", type=Path, default=Path(r"E:\diffusion_phase6a_titan\server_cache_v1"))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--transfer-workers", type=int, default=4)
    parser.add_argument("--attach", action="store_true", help="Resume delivery to an existing formal process without restarting inference")
    args = parser.parse_args()
    client = connect(args)
    sftp = client.open_sftp()
    args.cache.mkdir(parents=True, exist_ok=True)
    remote_cache = REMOTE+"/cache_v1"
    remote_report = SERVER_REPO+"/reports/phase6a_titan_semantic_preflight"
    small_files = ["scripts/phase6a_titan_semantic_preflight.py", "scripts/phase6a_titan_memory.py",
                   "scripts/summarize_phase6a_titan.py", "scripts/download_phase6a_titan.py", "scripts/phase6a_titan_remote.py",
                   "configs/phase6a_titan_semantic_preflight.json", "configs/phase6a_titan_crc_morphology_concepts.json",
                   "reports/phase6_conch_v15_download_audit/cohort_manifest.csv", "reports/phase6a_titan_semantic_preflight/README.md"]
    for relative in small_files:
        put(sftp, ROOT/relative, SERVER_REPO+"/"+relative)
    for path in args.model.iterdir():
        if not path.is_file():
            continue
        destination = REMOTE+"/model/"+path.name
        try:
            exists = sftp.stat(destination).st_size == path.stat().st_size
        except FileNotFoundError:
            exists = False
        if not exists:
            print("MODEL_TRANSFER", path.name, path.stat().st_size, flush=True)
            put(sftp, path, destination)
    # Clinical labels are never uploaded: use existing approved server data and
    # the experiment's explicit five-column reader (no survival columns).
    manifest = pd.read_csv(ROOT/"reports/phase6_conch_v15_download_audit/cohort_manifest.csv", dtype=str)
    mkdirs(sftp, remote_cache)
    mkdirs(sftp, REMOTE+"/staging")
    command = [REMOTE+"/env/bin/python", "-u", SERVER_REPO+"/scripts/phase6a_titan_semantic_preflight.py",
               "--stage", "all", "--features", REMOTE+"/staging", "--model", REMOTE+"/model",
               "--cache", remote_cache, "--clinical", CLINICAL, "--streamed-inputs", "--allocation-mode", "official"]
    bootstrap = "import subprocess,pathlib\n"+f"base=pathlib.Path({remote_cache!r})\n"+f"cmd={command!r}\n"+\
                "out=(base/'formal_run_stdout.log').open('wb');err=(base/'formal_run_stderr.log').open('wb')\n"+\
                "p=subprocess.Popen(cmd,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)\n"+\
                "(base/'formal_run.pid').write_text(str(p.pid));print('REMOTE_FORMAL_PID',p.pid)\n"
    if args.attach:
        with sftp.open(remote_cache+"/formal_run.pid") as stream:
            remote_pid = int(stream.read())
        print("ATTACHED_FORMAL_PID", remote_pid, flush=True)
    else:
        stdin, stdout, stderr = client.exec_command("/root/miniconda3/bin/python3 -")
        stdin.write(bootstrap)
        stdin.channel.shutdown_write()
        started_text = stdout.read().decode().strip()
        print(started_text, flush=True)
        error = stderr.read().decode()
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError(error)
        remote_pid = int(started_text.split()[-1])
    records = list(manifest.loc[manifest.status=="EXACT_SLIDE_MATCH"].sort_values(["patient_id", "filename"]).itertuples(index=False))

    def remote_health():
        command = f"ps -o stat= -p {remote_pid}"
        _, output, _ = client.exec_command(command)
        stat = output.read().decode().strip()
        if not stat or stat.startswith("Z"):
            try:
                sftp.stat(remote_report+"/DONE.json")
            except FileNotFoundError:
                with sftp.open(remote_cache+"/formal_run_stderr.log") as stream:
                    error = stream.read().decode(errors="replace")
                raise RuntimeError("Server formal process failed: "+error[-5000:])

    thread_clients = threading.local()
    upload_clients = []
    clients_lock = threading.Lock()

    def upload_slide(row):
        if not hasattr(thread_clients, "ssh"):
            thread_clients.ssh = connect(args)
            with clients_lock:
                upload_clients.append(thread_clients.ssh)
        with thread_clients.ssh.open_sftp() as transfer:
            destination = REMOTE+"/staging/"+row.delivery_relative_path
            completed = remote_cache+"/slides/"+Path(row.filename).stem+".npz"
            try:
                transfer.stat(completed)
                return
            except FileNotFoundError:
                pass
            try:
                if transfer.stat(destination).st_size == int(row.conch_size_bytes):
                    return
            except FileNotFoundError:
                pass
            put(transfer, args.features/row.delivery_relative_path, destination)

    started = time.monotonic()
    uploaded = 0
    with ThreadPoolExecutor(max_workers=args.transfer_workers) as pool:
        for batch_start in range(0, len(records), args.window):
            if batch_start:
                previous = records[max(0, batch_start-args.window):batch_start]
                for row in previous:
                    result = remote_cache+"/slides/"+Path(row.filename).stem+".npz"
                    check = 0
                    while True:
                        try:
                            sftp.stat(result)
                            break
                        except FileNotFoundError:
                            if check % 30 == 0:
                                remote_health()
                            check += 1
                            time.sleep(1)
            futures = [pool.submit(upload_slide, row) for row in records[batch_start:batch_start+args.window]]
            for future in futures:
                future.result()
                uploaded += 1
            print(f"STREAMED_H5 {uploaded}/{len(records)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    for upload_client in upload_clients:
        upload_client.close()
    print("ALL_INPUTS_DELIVERED; waiting for geometry and clinical gates", flush=True)
    while True:
        try:
            sftp.stat(remote_report+"/DONE.json")
            break
        except FileNotFoundError:
            remote_health()
            time.sleep(10)
    _, stdout, stderr = client.exec_command(shlex.join([REMOTE+"/env/bin/python", SERVER_REPO+"/scripts/summarize_phase6a_titan.py"]))
    print(stdout.read().decode(), flush=True)
    if stdout.channel.recv_exit_status() != 0:
        raise RuntimeError(stderr.read().decode())
    for filename in sftp.listdir(remote_report):
        sftp.get(remote_report+"/"+filename, str(ROOT/"reports/phase6a_titan_semantic_preflight"/filename))
    # Retrieve representation and OOF caches to local E drive; never put these
    # patient-level arrays in git.
    for filename in ["protocol_lock.json", "patient_representations.npz"]:
        sftp.get(remote_cache+"/"+filename, str(args.cache/filename))
    for folder in ["slides", "oof"]:
        (args.cache/folder).mkdir(exist_ok=True)
        try:
            names = sftp.listdir(remote_cache+"/"+folder)
        except FileNotFoundError:
            names = []
        for filename in names:
            sftp.get(remote_cache+"/"+folder+"/"+filename, str(args.cache/folder/filename))
    # Explicitly authorized cleanup: delete only this run's server weight file.
    weight_path = REMOTE+"/model/model.safetensors"
    if PurePosixPath(weight_path).parent != PurePosixPath(REMOTE+"/model"):
        raise RuntimeError("unexpected cleanup destination")
    sftp.remove(weight_path)
    with sftp.open(remote_report+"/DONE.json") as stream:
        done = json.load(stream)
    transfer = {"status": "PASS", "compute": "Server RTX 4080 SUPER 32 GB", "inputs_streamed": uploaded,
                "input_bytes": int(manifest.loc[manifest.status=="EXACT_SLIDE_MATCH","conch_size_bytes"].astype(int).sum()),
                "queue_window": args.window, "max_inflight_windows": 2, "transfer_workers": args.transfer_workers, "clinical_labels_uploaded": False,
                "server_weight_copy_removed": True, "large_source_files_retained_on_local_E": True}
    text = json.dumps(transfer, indent=2)+"\n"
    (ROOT/"reports/phase6a_titan_semantic_preflight/transfer_audit.json").write_text(text)
    with sftp.open(remote_report+"/transfer_audit.json", "w") as stream:
        stream.write(text)
    print("REMOTE_EXPERIMENT_FINISHED", done["status"], flush=True)
    sftp.close()
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
