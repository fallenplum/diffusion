"""Read/download official release assets into the data partition, no env overwrite."""
from pathlib import Path
import json,urllib.request,subprocess,sys,importlib.util
P=Path('/root/autodl-tmp/phase17_ssd_private');dest=P/'wheels';dest.mkdir(exist_ok=True)
out={}
for name,repo,tag in [('mamba_ssm','state-spaces/mamba','v2.2.4'),('causal_conv1d','Dao-AILab/causal-conv1d','v1.5.0.post8')]:
    url=f'https://api.github.com/repos/{repo}/releases/tags/{tag}'
    try:
        payload=json.load(urllib.request.urlopen(url,timeout=45))
        matches=[a for a in payload['assets'] if 'cp312' in a['name'] and 'torch2.5' in a['name'] and 'cu12' in a['name'] and 'abiFALSE' in a['name']]
        if len(matches)!=1:raise ValueError('exact compatible wheel count '+str(len(matches)))
        asset=matches[0];p=dest/asset['name']
        if not p.exists():urllib.request.urlretrieve(asset['browser_download_url'],p)
        out[name]={'status':'DOWNLOADED','url':asset['browser_download_url'],'bytes':p.stat().st_size}
        q=subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(P/'paper_deps'),str(p)],capture_output=True,text=True,timeout=120)
        out[name]['install_exit']=q.returncode;out[name]['install_log']=q.stdout[-600:]+q.stderr[-600:]
    except Exception as ex:out[name]={'status':'BLOCKED','error':repr(ex)}
    print(name,json.dumps(out[name]),flush=True)
(P/'PAPER_DEPENDENCIES.json').write_text(json.dumps(out,indent=2))
