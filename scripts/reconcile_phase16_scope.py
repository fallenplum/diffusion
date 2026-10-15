"""Apply the independently verified E-drive coverage correction without refitting."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase16_routing.core import *
def main():
    scope=json.loads((ROOT/'configs/phase16_scope_repair.json').read_text());write(R/'CANDIDATE_SCOPE.json',scope)
    for path in [P/'ASSET_SUMMARY.json',R/'G0_DECISION.json']:
        z=json.loads(path.read_text());z.update(candidate_pool_limited=True,original_cohort_slides=520,missing_original_CONCH_slides=1,available_original_H5_rows_complete=True);write(path,z)
    path=R/'DATA_INTERFACE_AUDIT.md';text=path.read_text();text+='\n\n## E盘重新接入后的范围修复\n\n**candidate-pool-limited=true**：519/520张原始切片有CONCH，覆盖512/512名患者。E盘全部CONCH及HF缓存检索未发现该队列额外文件。519个H5的所有原始patch行均恢复；一张原始UNI2切片缺少对应CONCH，同患者另一张切片已存在，不重命名/替换。最初将历史519个CONCH文件均恢复等同于完整原始WSI集合是范围标注错误，已记录BUG_LOG；不改变任何特征、训练、假设或预测。所有已训练arms统一使用这519张切片。\n';path.write_text(text,encoding='utf8')
    for path in (P/'cells').glob('*.json'):
        z=json.loads(path.read_text());z['candidate_pool_limited']=True;z['scope_correction_sha256']=sha(ROOT/'configs/phase16_scope_repair.json');write(path,z)
    print('CANDIDATE_SCOPE_CORRECTED_519_OF_520',flush=True)
if __name__=='__main__':main()
