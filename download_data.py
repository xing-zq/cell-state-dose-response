"""Download only numeric data, verify checksums, and never extract arbitrary ZIP paths."""
from pathlib import Path
import argparse,hashlib,urllib.request,zipfile,shutil

RX={
 'metadata.csv':('https://storage.googleapis.com/rxrx/rxrx19b/rxrx19b-metadata.zip','8cfba1c58a738a6bd0ff0af83efbe3d323cd90a17fada114792afb365f47be8f'),
 'embeddings.csv':('https://storage.googleapis.com/rxrx/rxrx19b/rxrx19b-dl-embeddings.zip','35672cd55acd817a57dda3fad30e19e0665be0728fb4c7387ce1a5cf90b478b6')}
OA={'metadata.parquet':'6731b56f8f4fe2db31fcdf1308c305fb','cellprofiler_raw.parquet':'0cf2b9d11268c363d756e69851a1a568'}
def digest(path,algorithm):
 h=hashlib.new(algorithm)
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def download(url,path):
 req=urllib.request.Request(url,headers={'User-Agent':'course-morphology-reanalysis/1.0'})
 with urllib.request.urlopen(req,timeout=120) as response,path.open('wb') as out:shutil.copyfileobj(response,out)
def main(dataset,root):
 root.mkdir(parents=True,exist_ok=True)
 for name,item in (RX if dataset=='rxrx19b' else OA).items():
  url,want=item if dataset=='rxrx19b' else (f'https://zenodo.org/api/records/17067683/files/{name}/content',item)
  algorithm='sha256' if dataset=='rxrx19b' else 'md5';target=root/name
  if target.exists():
   if digest(target,algorithm)==want:print('Verified existing',name);continue
   raise ValueError(f'{target} differs; preserve it and choose an empty --out directory.')
  temporary=root/(name+'.download');download(url,temporary)
  verified=root/(name+'.verified')
  if dataset=='rxrx19b':
   with zipfile.ZipFile(temporary) as z:
    matches=[n for n in z.namelist() if Path(n).name==name]
    if len(matches)!=1:raise ValueError(f'Unexpected ZIP contents for {name}')
    with z.open(matches[0]) as src,verified.open('wb') as dst:shutil.copyfileobj(src,dst)
   temporary.unlink()
  else:temporary.replace(verified)
  if digest(verified,algorithm)!=want:raise ValueError(f'Checksum mismatch for {name}; do not use changed data silently.')
  verified.replace(target);print('Downloaded and verified',name)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('dataset',choices=['rxrx19b','oasis']);p.add_argument('--out',type=Path);a=p.parse_args();main(a.dataset,a.out or Path(__file__).parent/'data'/a.dataset)
