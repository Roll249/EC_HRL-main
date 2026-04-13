# EC_HRL

## Environment setup (VS Code / Jupyter)

Lỗi `ModuleNotFoundError: No module named 'tensorflow.python'` thường do cài TensorFlow không đầy đủ hoặc Jupyter đang dùng **Python khác** với nơi bạn chạy `pip`. Nên dùng **virtual environment** và chọn đúng kernel.

### 1. Tạo và kích hoạt venv (Windows PowerShell)

```powershell
cd path\to\EC_HRL-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 2. VS Code / Jupyter

- `Ctrl+Shift+P` → **Python: Select Interpreter** → chọn `.venv\Scripts\python.exe`.
- Mở `Main_Simulation.ipynb` → chọn kernel trùng với interpreter đó (tên thường có `.venv`).

### 3. Nếu vẫn lỗi TensorFlow

Gỡ rồi cài lại trong **cùng** venv:

```powershell
python -m pip uninstall tensorflow tensorflow-intel -y
python -m pip cache purge
python -m pip install -r requirements.txt
```

### Đặt trọng số

Đặt các file `*.weights.h5` vào thư mục `weights` (cùng thư mục làm việc khi chạy notebook) hoặc set biến môi trường `EC_HRL_WEIGHTS_DIR` trỏ tới thư mục chứa weights.
