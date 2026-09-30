# Data

## `psds/`

Detector noise curves used by the two shipped cases.

| File | Used by | Format | Description |
|---|---|---|---|
| `aligo_O3actual_H1.txt` | `configs/eccentric_5d.py` | ASD | measured LIGO Hanford sensitivity, third observing run |
| `aLIGOZeroDetHighPower.txt` | `configs/aligned_3d.py` | PSD | Advanced LIGO design sensitivity, zero-detuning high-power |

Both are plain text with two columns (frequency in Hz, then ASD or PSD). A case
config states which of the two conventions its file uses, via the second
argument of `load_PSD`.

## Reproduce results 

```bash
python scripts/train_flow.py --config configs/eccentric_5d.py      

python scripts/generate_bank.py --config configs/eccentric_5d.py   
```


