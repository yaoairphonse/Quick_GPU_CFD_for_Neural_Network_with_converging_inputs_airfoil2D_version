# quick_CFD_for_NNLCI_airfoil_opt_case

**先開 [GUIDE.html](GUIDE.html)**：完整教學、方程式、演算法、網格預設、設定檔編輯器與輸出定義。

獨立 pipeline：**GXXX 點座標 CSV → 輪廓整理 → LF1／LF2 AMR cut-cell 製網 → GPU solver → SI 流場／壁面／載荷**。
此資料夾包含原始碼、獨立 CMake、Python 入口、預設設定與十份範例；不需引用父專案程式或幾何。

## 下次開發優先（2026-10-20）

使用者指定先進行 **低Mach preconditioning**，讓整體求解器與此pipeline支援低速；熱傳排在後續。
目前僅記錄，尚未有preconditioning開關或實作；詳見 [NEXT_SESSION.md](NEXT_SESSION.md)。

## 快速開始

需要 Python 3.9+、NumPy、CMake 3.18+、C++17；GPU 路徑另需 CUDA Toolkit 與相容 NVIDIA GPU。
目前預設 CUDA architecture=89（RTX4090）；其他 GPU 在 build 時明確指定架構。

```sh
cd quick_CFD_for_NNLCI_airfoil_opt_case
python3 -m pip install -r requirements.txt
python3 quick_cfd.py build
python3 quick_cfd.py run --points examples/G0001_airfoil_points.csv --out runs/my_airfoil
```

一次輸入，預設跑 **LF1 與 LF2**。輸出資料夾必須是新的或空的，避免覆寫既有算例。
每級各自均勻遠場 t=0 開始，沒有 LF1 → LF2 暖啟動。完成後開啟 `runs/my_airfoil/index.html`。

```sh
# 只製作兩級網格
python3 quick_cfd.py mesh --points examples/G0001_airfoil_points.csv --out runs/mesh_only
# 複製完整設定，再修改 JSON
python3 quick_cfd.py init-config --out my_config.json
python3 quick_cfd.py run --points examples/G0002_airfoil_points.csv --config my_config.json
# 物理遠場、laminar、常數黏度、局部輸出
python3 quick_cfd.py run --points examples/G0001_airfoil_points.csv --mach 1.35 --aoa 5 --pressure 18754 --temperature 216.65 --turbulence none --viscosity constant --region -0.5 2 -0.5 0.5 --fields x y p mach
# 只跑 LF2，或明確計算到指定 t*（不宣稱穩態）
python3 quick_cfd.py run --points examples/G0001_airfoil_points.csv --levels LF2 --mode time --stop-time 1
# 重新輸出，不跑 solver；範圍是 x/c,y/c，輸出欄位仍為 SI
python3 quick_cfd.py export --run runs/my_airfoil --fields x y p mach --region -0.5 2 -0.5 0.5 --formats npz
# CPU 可獨立建置（無需 CUDA）
python3 quick_cfd.py build --cpu
python3 quick_cfd.py run --backend cpu --points examples/G0001_airfoil_points.csv --mode time --stop-time 0.01
```

## 預設值

|設定|LF1|LF2|
|---|---|---|
|典型流體控制體|4668–4725|9433–9533|
|背景格數|42×40|42×40|
|最高加密層數|5|6|
|壁面 h/c|1/64|1/128|
|壁面 buffer|4|4|

實際格數隨外型改變；這兩個名稱代表固定規則，不是強制指定格數。完整 boxes 與參數在 [config/default.json](config/default.json)。

共同：M∞=1.65、AOA=1.5°、p∞=18754 Pa、T∞=216.65 K、chord=1 m；域 [-7,14]×[-10,10]（弦長單位）。
SA-neg、Sutherland、HLLC、一階 FV、SSP-RK3、CFL=.15、viscous/SA CFL=.1、GPU fused/Graph/batch32。
四面 characteristic farfield；可改 pressure-outlet／transmissive。壁面固定 no-slip；κ=0。

## 設定與輸出

- `freestream`：Mach、AOA、靜態 p/T、chord、γ、R。輸入點 x,y 為 x/c,y/c，不會擅自平移或縮放。
- `mesh`：bounds、base_shape、wall_buffer、max_cells、LF1/LF2 levels 與 boxes。
- `physics`：turbulence=`none|sa-neg`、viscosity=`constant|sutherland`、μ與Sutherland係數、SA遠場量。
- `boundaries`：xmin/xmax/ymin/ymax 各自設定；pressure-outlet 的靜態壓力用 Pa，supersonic outflow 不強制出口壓力。
- `solver`：GPU/CPU、hllc/ausm、reference/fused、CFL或固定dt（秒）、時間／預算／穩態條件、Mz參考點。
- `output`：x、y、p、rho、u、v、mach、temperature；CSV／未壓縮NPZ；範圍選擇、nearest-cell probes、表面patch、全翼型loads。

每級資料：`internal/`保存無因次NPY與幾何；`output/`保存選擇的SI資料；`result.json`、`solver.log`、`config.resolved.json`與`command.json`保存來源與停止原因。

`output/aerodynamics.csv/.npz`以每案一列輸出可選的drag、lift、fx、fy、viscous_force_x/y、viscous_drag/lift、moment_z、CL/CD/CM_z等；使用`--load-fields`或`output.load_fields`選欄位。

`output/loads.json`含 pressure／viscous／total 各自的 force_x、force_y、drag、lift、moment_z、CD、CL、CM_z，以及各patch分量。
**x–y 平面，Mz 逆時針正，預設參考點 (.25c,0)。力單位 N/m span，力矩 N m/m span。**
壁面壓力／密度／溫度取 owner cell；壁面速度=0。`temperature_owner_K`不是指定的壁溫。

退出碼：0=所要求的mesh/time任務完成或steady達標；2=輸入／建置／數值／輸出錯誤；3=steady未達標（仍保存結果）。
固定dt會取代自動convective／viscous／SA限制，不會自動把過大的dt縮成穩定值。

## 範圍與來源

這是快速、粗網格定性 aerodynamic 工具；mechanical SA-neg／laminar、κ=0，尚無Fourier、壁面模型或低Mach預條件。
溫度由總能量求得；尚未認證高Re邊界層／實驗CD。完整方法與限制見GUIDE。搬移資料夾後build會保留並改名舊位置的build目錄，再重新配置。
`CORE_PROVENANCE.json`記錄抽取時的原核心hash；父專案維持原樣。此獨立版修改介面與輸出，沒有改核心數值離散。

已交付的完整範例：[runs/G0001_default/index.html](runs/G0001_default/index.html)。
版本比對：[validation/ten_short_checks.json](validation/ten_short_checks.json)；完整稽核與計時見 [validation/summary.json](validation/summary.json)。
