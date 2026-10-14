"""Read-only cache and audited GPU preload: values/masks exactly preserved."""
import time,numpy as np,torch
from src.phase11_semantic.data import PatchCache


class CachedPatches(PatchCache):
    def __init__(self,directory,budget,preload=True):
        super().__init__(directory,budget);self.load_seconds=0.;self.calls=0;self.preload=preload and torch.cuda.is_available()
        if self.preload:
            start=time.monotonic();nslide=len(self.rows);self.slide_count=np.array([len(v) for v in self.by_patient])
            self.indices=np.full((512,int(self.slide_count.max())),nslide,int)
            arr=np.zeros((nslide+1,budget,768),np.float32);mask=np.zeros((nslide+1,budget),bool);self.counts=np.zeros(nslide+1,int)
            row_number={r['offset']:j for j,r in enumerate(self.rows)}
            for i,rows in enumerate(self.by_patient):
                for k,r in enumerate(rows):
                    j=row_number[r['offset']];self.indices[i,k]=j;n=min(budget,r['selected']);self.counts[j]=n
                    arr[j,:n]=self.arr[r['offset']:r['offset']+n];mask[j,:n]=True
            self.gpu_features=torch.tensor(arr,device='cuda');self.gpu_mask=torch.tensor(mask,device='cuda')
            torch.cuda.synchronize();self.preload_seconds=time.monotonic()-start
    def batch(self,ix,device):
        start=time.monotonic()
        if not self.preload or str(device)=='cpu':result=super().batch(ix,device)
        else:
            ix=np.asarray(ix,int);slides=int(self.slide_count[ix].max());indices=self.indices[ix,:slides];n=int(self.counts[indices].max())
            jj=torch.tensor(indices,device=device);result=self.gpu_features[:,:n,:][jj].contiguous(),self.gpu_mask[jj,:n].contiguous()
        self.load_seconds+=time.monotonic()-start;self.calls+=1
        return result
