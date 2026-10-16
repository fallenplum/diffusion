import json,subprocess,sys,time
from pathlib import Path
from src.trimodal_interaction.data import ROOT,P,R,cfg,source,write,digest

def main():
    begin=time.time();run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_trimodal_interaction.py'],cwd=ROOT,capture_output=True,text=True)
    if run.returncode:raise RuntimeError(run.stdout+run.stderr)
    assert '39 passed' in run.stdout
    write(R/'OPERATOR_AUDIT.json',{'status':'PASS','actual_test_process_exit':run.returncode,'stdout':run.stdout.strip(),'stderr':run.stderr.strip(),'tests':39,'tests_sha256':digest(ROOT/'tests/test_trimodal_interaction.py'),'training_source':source(),'seconds':time.time()-begin,'scope':'synthetic operator, gradient and fail-closed checks; not real-data biological efficacy'})
    base=cfg()['parent_commit'];changed=subprocess.check_output(['git','diff','--name-only',base],cwd=ROOT,text=True).splitlines()
    if changed:raise ValueError('inherited tracked files modified: '+repr(changed))
    m=json.loads((ROOT/'reports/phase17r_conditional_repair/ARTIFACT_MANIFEST.json').read_text())
    for row in m['public_files']:assert digest(ROOT/row['path'])==row['sha256'],row['path']
    write(R/'PARENT_IMMUTABILITY_AUDIT.json',{'status':'PASS','parent_commit':base,'parent_public_manifest_files_verified':len(m['public_files']),'all_inherited_tracked_files_unchanged':True,'training_source':source()})
    print('OPERATOR_PARENT_PASS',len(m['public_files']),flush=True)

if __name__=='__main__':main()
