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
    base = Path(raw_dir) if raw_dir is not None else Path(get_ds_par("incart", "raw_dir"))
    required_ext = [".dat", ".hea", ".atr"]
    for ext in required_ext:
        if not (base / f"{record_name}{ext}").exists():
            return False
    return True


def incart_raw_complete(raw_dir: Path | str, records: list[str] | None = None) -> bool:
    raw_path = Path(raw_dir)
    recs = records if records is not None else list(get_ds_par("incart", "records"))
    return all(is_record_downloaded(r, raw_path) for r in recs)


def sync_incart_with_gdrive(gdrive_data_raw: Path, *, max_workers: int = 6, verbose: bool = False) -> None:
    """
    Ha a ``gdrive_data_raw/incart`` alatt megvan az összes INCART rekord, bemásolja a
    projekt ``DATA_DIR/raw/incart`` mappájába.

    Ha nincs teljes másolat, ``download_incart_paralel()`` lefut, majd a helyi raw
    ``incart`` tartalma bemásolódik a Drive ``gdrive_data_raw/incart`` alá.
    """
    gdrive_raw = Path(gdrive_data_raw)
    gdrive_incart = gdrive_raw / "incart"
    local_incart = Path(get_ds_par("incart", "raw_dir"))
    records = list(get_ds_par("incart", "records"))

    local_incart.mkdir(parents=True, exist_ok=True)
    gdrive_incart.parent.mkdir(parents=True, exist_ok=True)

    def _mirror(src: Path, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    if incart_raw_complete(gdrive_incart, records):
        if verbose:
            print(f"[INCART] Drive-on lévő másolás: {gdrive_incart} -> {local_incart}")
        _mirror(gdrive_incart, local_incart)
    else:
        if verbose:
            print("[INCART] Nincs teljes adat a Drive-on – letöltés, majd mentés Drive-ra …")
        download_incart_paralel(max_workers=max_workers, verbose=verbose)
        if verbose:
            print(f"[INCART] Lokális -> Drive: {local_incart} -> {gdrive_incart}")
        _mirror(local_incart, gdrive_incart)


def download_record(record_name: str, verbose: bool = False):
    try:
        if is_record_downloaded(record_name):
            if(verbose):
                print(f"[SKIP] {record_name} already downloaded")
            return

        if(verbose):
            print(f"[DOWNLOAD] {record_name}")

        wfdb.dl_database(
            "incartdb",
            dl_dir=str(get_ds_par("incart", "raw_dir")),
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


def download_incart_paralel(max_workers: int = 6, verbose: bool = False) -> None:
    print("INCART parallel download started...")

    rawdir = Path(str(get_ds_par("incart", "raw_dir")))
    rawdir.mkdir(parents=True, exist_ok=True)
    records = get_ds_par("incart", "records")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(lambda r: download_record(r, verbose), records))

    print("Done.")


if __name__ == "__main__":
    download_incart_paralel()
