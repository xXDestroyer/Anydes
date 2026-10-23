import json, os, sys, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'anydes_tools/logs/phase_g1/candidate'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    from anydes_tools.tests.validate_candidate_phase_g import protected_hashes, sha
    mode=sys.argv[1]
    if mode=='before':
        snapshot=OUT/'snapshot'
        for rel in ('probar_anydes_tools_conversacion_v001_5.py','anydes_tools/runtime_environment.py'):
            dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        (OUT/'before.json').write_text(json.dumps(protected_hashes(),indent=2))
        print('G1 baseline captured')
    elif mode=='after':
        before=json.loads((OUT/'before.json').read_text());after=protected_hashes()
        changed=[p for p,h in before.items() if after.get(p)!=h]
        (OUT/'after.json').write_text(json.dumps({'changed':changed,'hashes':after},indent=2))
        print(json.dumps({'changed':changed}))
    elif mode=='gui':
        os.chdir(ROOT)
        os.environ['PYTHONDONTWRITEBYTECODE']='1'
        os.environ['ANYDES_RUNTIME_KIND']='CANDIDATE'
        os.environ['ANYDES_STAGING_DATA_ROOT']=str(OUT)
        import probar_anydes_tools_conversacion_v001_5 as bootstrap
        try:
            return bootstrap.main()
        except BaseException as e:
            (OUT/'startup_error.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)[:240]}))
            raise
if __name__=='__main__':
    raise SystemExit(main() or 0)

