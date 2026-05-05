#!/usr/bin/env python3

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import wfdb

from src.config import get_ds_par

def is_record_downloaded(record_name: str) -> bool:
    """
    Ellenőrzi, hogy minden szükséges fájl megvan-e.
    """
    required_ext = [".dat", ".hea", ".atr"]

    for ext in required_ext:
        if not (get_ds_par("incart", "raw_dir") / f"{record_name}{ext}").exists():
            return False

    return True


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
