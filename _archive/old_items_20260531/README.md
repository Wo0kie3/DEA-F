# Archived old repo items

This folder contains files and directories that were not required by methods 1-5.

The original relative paths are preserved under `_archive/old_items_20260531`, and
`restore_map.csv` maps each archived item back to its previous location.

To restore the previous layout from the repository root:

```powershell
.\_archive\old_items_20260531\restore_archived_items.ps1
```

The restore script stops if a destination path already exists, so it will not
silently overwrite the cleaned method 1-5 layout.
