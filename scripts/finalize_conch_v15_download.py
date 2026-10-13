"""Legacy Phase6 finalizer preserved privately because it embeds a case ID."""
import os,runpy
from pathlib import Path
p=Path(os.environ.get('PHASE6_PRIVATE_FINALIZER','/root/phase10_trimodal_private/inherited_identifiable_originals/scripts/finalize_conch_v15_download.py'))
if not p.is_file():raise RuntimeError('Set PHASE6_PRIVATE_FINALIZER to the authorized private original; historical Git ref is unchanged')
if __name__=='__main__':runpy.run_path(str(p),run_name='__main__')
