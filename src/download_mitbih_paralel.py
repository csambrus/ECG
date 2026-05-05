#!/usr/bin/env python3

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil
import wfdb

from src.config import get_ds_par

def is_record_downloaded(record_name: str, raw_dir: Path | str | None = None) -> bool:
    """
    Ellenőrzi, hogy minden szükséges fájl megvan-e.
    """
    base = Path(raw_dir) if raw_dir is not None else Path(get_ds_par("mitbih", "raw_dir"))
    required_ext = [".dat", ".hea", ".atr"]
    for ext in required_ext:
        if not (base / f"{record_name}{ext}").exists():
            return False
    return True


def mitbih_raw_complete(raw_dir: Path | str, records: list[str] | None = None) -> bool:
    raw_path = Path(raw_dir)
    recs = records if records is not None else list(get_ds_par("mitbih", "records"))
    return all(is_record_downloaded(r, raw_path) for r in recs)


def sync_mitbih_with_gdrive(gdrive_data_raw: Path, *, verbose: bool = False) -> None:
    """
    Ha a ``gdrive_data_raw/mitbih`` alatt megvan az összes MIT-BIH rekord, bemásolja a
    projekt ``DATA_DIR/raw/mitbih`` mappájába.

    Ha nincs teljes másolat, ``download_mitbih_paralel()`` lefut, majd a helyi raw
    ``mitbih`` tartalma bemásolódik a Drive ``gdrive_data_raw/mitbih`` alá.
    """
    gdrive_raw = Path(gdrive_data_raw)
    gdrive_mitbih = gdrive_raw / "mitbih"
    local_mitbih = Path(get_ds_par("mitbih", "raw_dir"))
    records = list(get_ds_par("mitbih", "records"))

    local_mitbih.mkdir(parents=True, exist_ok=True)
    gdrive_mitbih.parent.mkdir(parents=True, exist_ok=True)

    def _mirror(src: Path, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    if mitbih_raw_complete(gdrive_mitbih, records):
        if verbose:
            print(f"[MITBIH] Drive-on lévő másolás: {gdrive_mitbih} -> {local_mitbih}")
        _mirror(gdrive_mitbih, local_mitbih)
    else:
        if verbose:
            print("[MITBIH] Nincs teljes adat a Drive-on – letöltés, majd mentés Drive-ra …")
        download_mitbih_paralel(verbose=verbose)
        if verbose:
            print(f"[MITBIH] Lokális -> Drive: {local_mitbih} -> {gdrive_mitbih}")
        _mirror(local_mitbih, gdrive_mitbih)


def download_record(record_name: str, verbose: bool = False):
    try:
        if is_record_downloaded(record_name):
            if(verbose):
                print(f"[SKIP] {record_name} already downloaded")
            return

        if(verbose):
            print(f"[DOWNLOAD] {record_name}")
        wfdb.dl_database(
            "mitdb",
            dl_dir=str(get_ds_par("mitbih", "raw_dir")),
            records=[record_name],
        )

        if is_record_downloaded(record_name):
            if(verbose):
                print(f"[OK] {record_name}")
        else:
            print(f"[WARN] {record_name} incomplete download")

    except Exception as e:
        print(f"[ERROR] {record_name}: {e}")
        raise


def download_mitbih_paralel(verbose: bool = False) -> None:
    print("MIT_BIH parallel download started...")
    
    raw_dir = Path(str(get_ds_par("mitbih", "raw_dir")))
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = get_ds_par("mitbih", "records")

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda r: download_record(r, verbose), records))

    print("Done.")

if __name__ == "__main__":
    download_mitbih_paralel()