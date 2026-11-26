The local archive is expected at `archive/` with `train/`, `val/`, and
`test/` folders, each containing `NORMAL/` and `PNEUMONIA/` classes. The
loader combines these folders and makes deterministic source-group splits
because related image variants and patient identifiers cross the supplied
splits. See `../DATASET_CITATION.md` for source and license details.

Do not commit or redistribute the image files without confirming the
applicable rights.
