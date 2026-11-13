# minimal stub: OCHRE imports tkinter only for its optional GUI
class _Any:
    def __getattr__(self, k): return _Any()
    def __call__(self, *a, **k): return _Any()
import sys
for n in ["Tk","Frame","Label","Button","Entry","StringVar","IntVar","Toplevel","filedialog","messagebox","ttk","END","W","E","N","S"]:
    globals()[n]=_Any()
