#!/usr/bin/env python3
"""Generate the offline teaching guide and editable configuration panel."""
from pathlib import Path
import html,json
R=Path(__file__).resolve().parents[1]
c=json.loads((R/'config/default.json').read_text())
body=r'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quick CFD for NNLCI：airfoil pipeline 教學</title>
<style>:root{--ink:#183747;--green:#087c68}body{font:16px/1.85 system-ui;color:var(--ink);background:#f1f5f7;margin:auto;max-width:1220px;padding:24px}h1,h2,h3{line-height:1.4}h2{color:var(--green)}section{background:white;border-radius:12px;padding:26px;margin:20px 0}nav{display:flex;gap:14px;flex-wrap:wrap}a{color:#006b9c}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce4e7;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f5;border-radius:8px;padding:16px;line-height:1.6}code{background:#edf3f5;padding:2px 5px}.eq{font:16px/1.9 ui-monospace,monospace;white-space:pre-wrap}.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.step{background:#e2f2ed;padding:14px;border-radius:8px}.note{background:#fff3db;padding:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{width:100%;background:white}label{display:inline-flex;gap:6px;align-items:center;margin:6px}input,select,button{font:inherit;padding:5px}input{width:100px}button{cursor:pointer}textarea{box-sizing:border-box;width:100%;min-height:360px;font:14px/1.5 ui-monospace,monospace;padding:14px}.scroll{overflow:auto}.pill{font-weight:700;color:var(--green)}@media(max-width:750px){.steps,.grid{grid-template-columns:1fr}section{padding:18px}}</style>
<h1>Airfoil points → LF1／LF2 → Quick CFD → NNLCI 資料</h1>
<p>獨立資料夾：<code>quick_CFD_for_NNLCI_airfoil_opt_case</code> · x–y 平面 · Mz 逆時針正 · 2026-10-20</p>
<nav><a href="#start">開始使用</a><a href="#input">輸入</a><a href="#mesh">製網</a><a href="#equations">方程式</a><a href="#scheme">數值方法</a><a href="#wall">壁面／邊界</a><a href="#sa">SA</a><a href="#time">步長／收斂</a><a href="#output">輸出／積分</a><a href="#editor">設定編輯器</a><a href="#roadmap">下一次開發</a><a href="#validation">驗證</a></nav>
<section id="start"><h2>1. 先跑一個案例</h2>
<div class="steps"><div class="step"><b>① 輸入</b><br>GXXX x,y CSV<br>只重排點，不移動外型</div><div class="step"><b>② Quick meshing</b><br>LF1／LF2<br>AMR＋cut-cell＋保守合併</div><div class="step"><b>③ Solver</b><br>HLLC／SSP-RK3<br>laminar或SA-neg</div><div class="step"><b>④ Output</b><br>SI場量／壁面／載荷<br>CSV與未壓縮NPZ</div></div>
<pre>cd quick_CFD_for_NNLCI_airfoil_opt_case
python3 -m pip install -r requirements.txt
python3 quick_cfd.py build
python3 quick_cfd.py run --points examples/G0001_airfoil_points.csv --out runs/my_airfoil</pre>
<p>開啟 <code>runs/my_airfoil/index.html</code>。一次輸入預設跑LF1、LF2；每級獨立均勻t=0，沒有coarse warm start。預先完成的範例：<a href="runs/G0001_default/index.html">G0001 default</a>。</p>
<p>軟體：Python3.9+／NumPy／CMake3.18+／C++17；GPU另需CUDA Toolkit及相容NVIDIA GPU。預設arch89對應本次RTX4090；其他GPU用 <code>python3 quick_cfd.py build --cuda-arch 86</code> 等明確設定。這不是所有GPU的通用binary。</p>
<pre># 無CUDA的CPU建置
python3 quick_cfd.py build --cpu
python3 quick_cfd.py run --backend cpu --points examples/G0001_airfoil_points.csv --mode time --stop-time .01
# 只製網；仍產生LF1/LF2與幾何table
python3 quick_cfd.py mesh --points examples/G0001_airfoil_points.csv --out runs/mesh_only
# 完整JSON設定，或直接用部分設定覆蓋預設
python3 quick_cfd.py init-config --out my_config.json
python3 quick_cfd.py run --points examples/G0001_airfoil_points.csv --config my_config.json</pre>
<p>資料夾搬位置後再build時，入口會將舊位置的CMake build目錄改名保存，再重新配置；不依賴父專案的build。歷史command.json內的絕對路徑只是當次來源紀錄。</p>
<p>輸出目錄須為新的或空目錄；不默默覆寫先前案例。<code>run</code>同一求解程序內製網後直接使用該mesh，不需要重建一次。<code>mesh</code>是獨立的幾何預覽；目前沒有讀取mesh-only結果直接續算的mesh-cache loader，之後另執行run會重新製網。</p>
</section>
<section id="input"><h2>2. 點座標格式與幾何處理</h2>
<pre>x,y
3.605044470E-01,-3.990721966E-02
...</pre>
<p>使用與提供的GXXX系列相同的格式：第一列x,y；兩欄有限數值；x/c介於0與1；前緣x=0只有一點；各個內部x有上下表面各一點；x=1含至少兩個不同的垂直尾緣點。列的順序可不沿輪廓，例如原輸入依y排序。檔名可用GXXXX_airfoil_points.csv；數值內容才是判斷依據。</p>
<p><code>pipeline/geometry.py</code>依相同x配對，上表面由TE走向LE，下表面由LE走向TE，再接原有垂直尾緣。所有座標都保留，不平滑、不重新取樣、不封掉原幾何資訊；重複點、缺少配對或不符此格式會明確拒絕。一般任意散點、NACA閉合重複點格式或尖尾緣檔案需要先轉換格式，這個入口不會猜測拓撲。</p>
<p>原始點是<b>x/c、y/c</b>。<code>chord_m</code>決定物理尺寸；例如chord=.5m時x/c=1對應.5m。改AOA會旋轉遠場速度，幾何與網格保持原座標。patch標記：1=upper、2=lower、3=vertical trailing edge。每條切割壁段都保存原source_edge與patch。</p>
<p>輸入副本、重排polygon、patches及SHA256保存在每次run的<code>geometry/</code>。</p>
</section>
<section id="mesh"><h2>3. 參數化quick meshing：兩套既有規則</h2>
<p>靜態quadtree Cartesian網格從矩形背景格開始。某格符合壁面距離條件，或與指定box相交且還沒到box要求的level，就分成4個子格。之後做相鄰格2:1平衡；粗細交界一個粗面對兩個子面，為每個實際subface建立共享通量。這是static AMR；求解中不動態改網格，也沒有local subcycling。</p>
<div class="scroll"><table><tr><th>參數</th><th>LF1</th><th>LF2</th><th>意義</th></tr><tr><td>bounds_xy_c</td><td colspan="2">[-7,14,-10,10]</td><td>x/c、y/c界限</td></tr><tr><td>base_shape</td><td colspan="2">[42,40]</td><td>每個背景格.5c×.5c</td></tr><tr><td>levels</td><td>5</td><td>6</td><td>0為背景層，最高細化層</td></tr><tr><td>wall_buffer</td><td colspan="2">4</td><td>用在下面的幾何距離準則</td></tr><tr><td>最小h/c</td><td>1/64</td><td>1/128</td><td>本預設正方形背景格</td></tr><tr><td>實測流體CV</td><td>4668–4725</td><td>9433–9533</td><td>隨外型變；非強制格數目標</td></tr></table></div>
<pre class="eq">h_x(level) = (xmax − xmin) / nx / 2^level
h_y(level) = (ymax − ymin) / ny / 2^level
壁面細化條件：distance(rectangle, polygon) ≤ buffer × 0.5 × min(h_x,h_y)
此外：若 rectangle 與 box 相交，且 level &lt; box.level，亦細化。</pre>
<p>buffer=4在這個實作中對應目前格尺度的2h距離判斷，不是固定物理厚度4c。保持足夠buffer使cut-cell留在同層細網格，避免複雜切割面直接接粗細交界。</p>
<table><tr><th>box [xmin,xmax,ymin,ymax,level]</th><th>LF1</th><th>LF2</th></tr><tr><td>近翼型／尾緣</td><td>[-.5,2,-.75,.75,3]</td><td>[-.5,2,-.75,.75,4]</td></tr><tr><td>尾流</td><td colspan="2">[1,4,-.35,.65,3]</td></tr><tr><td>中場</td><td colspan="2">[-2,6,-3,3,1]</td></tr><tr><td>額外細尾流</td><td>無</td><td>[2,3.25,-.35,.65,4]</td></tr></table>
<div class="grid"><div><h3>LF1實際背景member近景</h3><img src="docs/assets/LF1_mesh.svg" alt="LF1實際網格"></div><div><h3>LF2實際背景member近景</h3><img src="docs/assets/LF2_mesh.svg" alt="LF2實際網格"></div></div>
<p>圖中顯示背景member矩形加原始外型遮罩；不是精確cut-cell流體多邊形圖。積分使用下述真實幾何。</p>
<h3>3.1 Cut-cell的幾何積分</h3><pre class="eq">V_i = ∫_(cell ∩ fluid) dA
x̄_i = (1/V_i) ∫_(cell ∩ fluid) x dA
每個Cartesian面：只保留實際流體開口長度 A_f 與其重心。
每個壁段：保留兩端點、長度 ℓ_w、流體指向固體的單位法向 n_w、patch、source_edge。
幾何閉合：Σ_f A_f n_if + Σ_w ℓ_w n_w = 0</pre>
<p>同一背景格若有兩塊不連通流體，建立不同CV；相連性由實際開口決定。網格線恰與垂直尾緣重合時會封住正確aperture並只歸屬流體側，不平移外型。</p>
<h3>3.2 小體積控制體的保守合併</h3><p>流體面積小於所在背景格面積一半的控制體，優先經最大正開口面與同層相連流體CV合併，直到達到幾何目標。內部面移除，壁段完整保留，V與一階面積矩相加；因此沒有跨固體混合。若無合法鄰居則報錯。</p>
<pre class="eq">V_merge = Σ_i V_i
x̄_merge = (Σ_i V_i x̄_i) / V_merge
內部共享面：F_f A_f − F_f A_f = 0</pre>
<p>此pipeline固定使用agglomerated FV／真實流體體積CFL；不是逐步state-SRD。只用「被合併後的外框矩形面積」代替V會破壞方法，所以輸出另留member映射。</p>
<p><code>mesh.max_cells</code>限制背景細化規模；預設200萬是保護上限，不代表會產生200萬格。改大levels／boxes會增加計算，現有4500/9000級的經驗時間不能直接外推。</p>
</section>
<section id="equations"><h2>4. 可壓縮方程與物理單位</h2>
<h3>4.1 平均流守恆式</h3><pre class="eq">U = [ρ, ρu, ρv, ρE]ᵀ
∂U/∂t + ∂F_c/∂x + ∂G_c/∂y = ∂F_v/∂x + ∂G_v/∂y
F_c = [ρu, ρu²+p, ρuv, u(ρE+p)]ᵀ
G_c = [ρv, ρuv, ρv²+p, v(ρE+p)]ᵀ
p = (γ−1) [ρE − ½ρ(u²+v²)]
T = p/(ρR),    a = √(γp/ρ),    Mach = √(u²+v²)/a</pre>
<p>以density-based形式直接更新守恆變數；壓力與溫度由EOS恢復。T∞、p∞都是靜態遠場量。AOA α定義u∞=Ma∞cosα、v∞=Ma∞sinα；不是旋轉輸出座標。</p>
<h3>4.2 Newtonian應力與黏性功</h3><pre class="eq">μ_eff = μ + μ_t
τ_xx = μ_eff [2u_x − ⅔(u_x+v_y)]
τ_yy = μ_eff [2v_y − ⅔(u_x+v_y)]
τ_xy = μ_eff (u_y+v_x)
F_v = [0, τ_xx, τ_xy, uτ_xx+vτ_xy]ᵀ
G_v = [0, τ_xy, τ_yy, uτ_xy+vτ_yy]ᵀ</pre>
<p>laminar時μ_t=0。SA-neg啟用後μ_t同時進入動量和能量黏性功。當前κ=0：<b>沒有Fourier分子／湍流熱通量</b>，因此這是mechanical viscous版本；T仍會隨壓縮、膨脹與黏性功改變，並非等溫流。</p>
<h3>4.3 Sutherland與常數黏度</h3><pre class="eq">μ(T) = μ₀ (T/T₀)^(3/2) (T₀+S)/(T+S)
μ₀=1.716×10⁻⁵ Pa·s, T₀=273.15 K, S=110.4 K</pre>
<p>這組SI常數沿用本專案。<code>viscosity=sutherland</code>每格、每RK stage更新；<code>constant</code>固定在T∞的Sutherland值，或使用<code>constant_mu_Pa_s</code>。SA開關與Sutherland開關彼此獨立。氣體黏度的溫度依賴背景可參考<a href="https://www.grc.nasa.gov/www/k-12/airplane/viscosity.html">NASA Glenn</a>；該頁使用Rankine，這裡設定檔使用K。</p>
<h3>4.4 內部無因次化，對外SI</h3><pre class="eq">ρ_ref = p∞/(R T∞), a_ref=√(γ R T∞), L_ref=c
x*=x/c, ρ*=ρ/ρ_ref, u*=u/a_ref
p*=p/(ρ_ref a_ref²), T*=T/(γT∞), μ*=μ/(ρ_ref a_ref c)
t*=t a_ref/c,   dt*=dt_seconds a_ref/c
p∞*=T∞*=1/γ, R*=1
S*=S/(γT∞), μ_ref*=μ∞/(ρ∞a∞c), T_ref*=1/γ</pre>
<p>預設a∞≈295.068m/s、ρ∞≈.301562731kg/m³、V∞≈486.862m/s、q∞≈35740.4355Pa，Re_c≈1.033×10⁷。原點座標和mesh bounds以弦長為單位；<code>output</code>的p、T、ρ、u、v、x、y會換回SI，<code>internal</code>的NPY保持無因次。</p>
</section>
<section id="scheme"><h2>5. 離散、scheme與預組裝table</h2>
<h3>5.1 Finite volume殘差</h3><pre class="eq">dU_i/dt = L_i(U)
L_i = −(1/V_i) [Σ_f A_f F̂_c(U_L,U_R;n_if)
                       −Σ_f A_f F̂_v + Σ_w 壁面積分]
共享面只算一次，左右CV以相反符號gather同一通量。</pre>
<p>對流為piecewise constant一階空間重建，沒有MUSCL斜率；一階不代表任意dt都穩定。黏性需要梯度，使用預組裝weighted least-squares。</p>
<h3>5.2 HLLC（預設）</h3><pre class="eq">S_L = min(u_L−a_L, ũ−ã),  S_R = max(u_R+a_R, ũ+ã)
S_* = [p_R−p_L+ρ_Lu_L(S_L−u_L)−ρ_Ru_R(S_R−u_R)]
      / [ρ_L(S_L−u_L)−ρ_R(S_R−u_R)]
F̂ = F_L                         if 0 ≤ S_L
     F_L + S_L(U_*L−U_L)         if S_L ≤ 0 ≤ S_*
     F_R + S_R(U_*R−U_R)         if S_* ≤ 0 ≤ S_R
     F_R                         if S_R ≤ 0</pre>
<p>u_L/u_R在上式是面法向速度；切向動量也保留。Roe平均只用於波速估計；這個HLLC通量不求全域反矩陣。核心在<code>src/flux.h</code>與CUDA對應實作。來源：<a href="https://doi.org/10.1007/BF01414629">Toro、Spruce、Speares，1994</a>。</p>
<details><summary>展開：AUSM+-up在此程式中的通量結構</summary><pre class="eq">â=(a_L+a_R)/2,  M_L=u_L/â, M_R=u_R/â
M_f=M₄⁺(M_L)+M₄⁻(M_R) − 壓力差修正
p_f=P₅⁺(M_L)p_L+P₅⁻(M_R)p_R − 速度差修正
ṁ=â M_f ρ_upwind
F̂=[ṁ, ṁu_upwind+p_f, ṁv_upwind, ṁH_upwind]ᵀ
本core：Kp=.25, Ku=.75, σ_AUSM=1, β=1/8, Mco=1。</pre><p>分裂多項式與修正項完整列在src/flux.h；Mco沿用可壓縮基準值1，這個入口未提供all-speed低Mach調整。</p></details>
<p>可選<code>solver.flux=ausm</code>，對應現有AUSM+-up分裂質量／壓力通量。它不是implicit求解器，也不會因此自動具備低Mach預條件；目前預設及基準仍採HLLC。</p>
<h3>5.3 固定LS矩陣、CSR與黏性面修正</h3><pre class="eq">min_g Σ_j w_j [g·(x_j−x_i) − (φ_j−φ_i)]²
M_i = Σ_j w_j d_ij d_ijᵀ
g_i = M_i⁻¹ Σ_j w_j d_ij (φ_j−φ_i)
面梯度：g_f = ½(g_L+g_R)
         + {[(φ_R−φ_L) − ½(g_L+g_R)·d] / |d|²} d</pre>
<p>固定幾何的2×2矩陣逆、weights、鄰居索引、face偏移、wall donor/distance、SA真實壁距都在setup預算，整理為CSR上傳。每步只做係數乘加，不反覆建矩陣。面修正讓線性場沿兩重心方向滿足差值。</p>
<h3>5.4 SSP-RK3時間更新</h3><pre class="eq">U¹ = Uⁿ + Δt L(Uⁿ)
U² = ¾Uⁿ + ¼[U¹ + Δt L(U¹)]
Uⁿ⁺¹ = ⅓Uⁿ + ⅔[U² + Δt L(U²)]</pre>
<p>每步三次RHS；整體空間精度仍受一階對流與壁面重建限制。時間方法來源：<a href="https://www.sciencedirect.com/science/article/pii/0021999188901775">Shu &amp; Osher，JCP 77，1988</a>。使用RK3不代表全套cut-wall方案已被證明三階。</p>
<h3>5.5 GPU執行方式</h3><p>狀態為FP64 SoA，常駐GPU；共享face flux後以cell CSR gather，沒有對整個稀疏矩陣求逆。CUDA Graph每批派發32個trial；這是連續時間步，<b>不是32個工況</b>。SA fused合併velocity/SA梯度、face-wall與gather重複掃描，現有agglomerated case每步Graph29→20個kernel；reference仍可選。這屬執行流程重構，不改物理或CFL。</p>
<p>此pipeline沒有新增implicit、local time stepping、multigrid、multi-case batch或shared-memory tiling。效能範圍分別記錄mesh、event、advance wall、process與Python export。</p>
</section>
<section id="wall"><h2>6. 壁面、角度、inlet與outlet</h2>
<h3>6.1 No-slip cut wall</h3><pre class="eq">u_wall=v_wall=0; ν̃_wall=0; μ_t,wall=0
(∂u/∂n)_wall ≈ (u_wall−u_donor)/d_n
(∂v/∂n)_wall ≈ (v_wall−v_donor)/d_n</pre>
<p>用實際可見的流體側donor，預設要求d_n≥.25h；若owner不合適則選合法donor，不把負距離裁成小正數。壁面應力使用分子μ，速度梯度為一側法向重建。固定壁面黏性功為0；κ=0下沒有定溫／絕熱Fourier壁面選項，<code>temperature_owner_K</code>只代表近壁取樣。</p>
<h3>6.2 Characteristic farfield（預設四面）</h3><pre class="eq">λ = [u_n−a, u_n, u_n, u_n+a]
超音速入流 u_n ≤ −a：遠場全部狀態進入。
超音速出流 u_n ≥  a：全部特徵離開，不強制出口壓力。
次音速：結合內部 outgoing 和遠場 incoming acoustic invariant，
         entropy／切向量由入流或出流方向選取。</pre>
<p>以每個面自己的外法向分類，AOA改變後各邊的入／出流由局部法向決定。上、下邊不能一律當固定入口或出口。遠場值用Mach、AOA、靜態p/T計算。</p>
<h3>6.3 可選pressure-outlet</h3><p>例如設定<code>boundaries.xmax="pressure-outlet"</code>與<code>outlet_pressure_Pa</code>。次音速向外流動時使用指定靜態p，保留相容的內部特徵；超音速向外流動時仍外推，p_out不強制。若回流，使用已設定的遠場reference作characteristic backflow reservoir；SA入流ν̃取遠場值、出流外推。實作見<code>src/static_amr2d_boundary.h</code>。</p>
<pre>{"boundaries": {"xmax":"pressure-outlet", "outlet_pressure_Pa":18754}}
python3 quick_cfd.py run --points examples/G0001_airfoil_points.csv --config config/pressure_outlet.json</pre>
<p><code>transmissive</code>可明確選擇，但預設外流場使用characteristic farfield。此介面沒有開放total-pressure inlet；core中雖有相關算子，若要擴充本pipeline需再設計與驗證參數接口。</p>
</section>
<section id="sa"><h2>7. 一方程SA-neg：變數與離散</h2>
<p>第五個守恆量Q=ρν̃。ν̃不是ν_t；ν_t由模型函數換算。依<a href="https://tmbwg.github.io/turbmodels/spalart.html">NASA TMR的SA-neg定義</a>，保留ft2與Note1c修正，未加入旋轉／粗糙度／額外compressibility correction。</p>
<pre class="eq">χ=ν̃/ν,  f_v1=χ³/(χ³+c_v1³),  μ_t=ρν̃ f_v1（ν̃≥0）
ν=μ/ρ,  c_v1=7.1, σ=2/3, c_b1=.1355, c_b2=.622, κ_SA=.41
∂Q/∂t + ∇·(ρuν̃) = ∇·(ρ A ∇ν̃)
                       + ρ[P−D+(c_b2/σ)|∇ν̃|²] − A∇ρ·∇ν̃
A=(ν+ν̃)/σ （非負分支）
P=c_b1(1−f_t2) S̃ ν̃
D=[c_w1 f_w − c_b1 f_t2/κ_SA²](ν̃/d)²
f_t2=1.2 exp(−.5χ²), c_w1=c_b1/κ_SA²+(1+c_b2)/σ</pre>
<p>密度梯度修正 −A∇ρ·∇ν̃ 是把同一ν̃擴散式轉成ρν̃守恆形式時需要的代數項，不是另外一個SA壓縮性修正模型。SA對流使用平均流同一mass flux，對ν̃做upwind；擴散採共享face通量與梯度修正。</p>
<details><summary>展開：與程式對應的SA閉合函數與負值分支</summary><pre class="eq">Ω=|v_x−u_y|, f_v2=1−χ/(1+χ f_v1), S̄=ν̃ f_v2/(κ_SA²d²)
S̃ = Ω+S̄                          if S̄ ≥ −.7Ω
   = Ω(−.01Ω−.1S̄)/(−.5Ω−S̄)    otherwise
r=min[ν̃/(S̃κ_SA²d²),10]；S̃=0時程式取r=10。
g=r+.3(r⁶−r), f_w=g[65/(g⁶+64)]^(1/6)
負ν̃分支：χ=ν̃/ν, f_n=(16+χ³)/(16−χ³)
A=(ν+ν̃ f_n)/σ, μ_t=0
模型源項 = c_b1(1−1.2)Ων̃ + c_w1(ν̃/d)²
密度與梯度項仍按上面的守恆換式處理。</pre><p>分段S̃公式採已驗證的代數等價寫法，避免極小Ω的中間乘積下溢。這個模型指定的r上界與SA-neg負分支，不等於裁切守恆狀態或人工限制μ_t。</p></details>
<p>負ν̃時使用SA-neg指定的source／diffusion延拓，μ_t=0，不把ν̃硬裁成0，也不設渦黏度硬上限。完整分支、f_w與S̃的Note1c公式就在隨附<code>src/sa_model.h</code>；guide不把標準SA方程誤稱為每個負值分支都相同。</p>
<p>d是CV重心到原始polygon線段或端點的真實最短距離，預算一次。這個d與壁面應力donor法向距離d_n不同。遠場預設ν̃∞=3ν∞，壁面ν̃=0；模型與源項限制也參與自動dt。</p>
<p class="note">SA-neg是已實作的mechanical coupling；目前沒有Fourier／湍流熱傳、wall function或NASA平板Cf/profile認證。Re≈10⁷與h≈c/128不足以保證壁層解析，CL/CD/CM應作定性或版本比較。</p>
</section>
<section id="time"><h2>8. CFL、固定dt、穩態條件</h2>
<h3>8.1 自動全域步長（預設）</h3><pre class="eq">Δt_conv,i = CFL × V_i / [Σ_f (|u_n|+a)_f A_f + 壁面acoustic rate]
Δt_visc,i = viscous_CFL × V_i / Σ_f [(4/3)ν_eff/d_f] A_f
Δt_SA,i   = SA_CFL / [2 Σ_f K_f /(ρ_i V_i) + SA_source_rate_i]
Δt = min_i(Δt_conv,i, Δt_visc,i, Δt_SA,i, 到本次目標時間的剩餘量)</pre>
<p>K_f是SA共享擴散係數乘面長除距離；壁面也納入。仍是explicit global stepping。預設CFL=.15、viscous=.1、SA=.1；可用JSON分別設定。<code>--cfl</code>會選自動CFL模式。</p>
<h3>8.2 固定物理dt</h3><pre>python3 quick_cfd.py run --points examples/G0001_airfoil_points.csv --mode time --stop-time .01 --dt 1e-7
# 或
{"solver":{"time_step":{"mode":"fixed","dt_seconds":1e-7}}}</pre>
<p>dt以秒輸入，轉成dt*；固定dt會取代自動convective／viscous／SA限制，僅在目標時間前截短。若過大，可能正性失敗；程式不為通過而調CFL、裁切狀態或偷偷換模型。<code>--cfl</code>與<code>--dt</code>不可同時指定。</p>
<h3>8.3 到時間與達穩態是不同任務</h3><p><code>--mode time --stop-time 1</code>計算到t*=1，不以達穩態為前提；<code>--physical-time .003</code>改用秒指定終點。兩個時間cap選項不可同時使用。預設<code>mode=steady</code>，time cap t*=100，每級118秒程序預算。</p>
<pre class="eq">近場Ω：x/c∈[−1,3]、y/c∈[−1,1]
R_k = Σ_(i∈Ω) V_i |L_i,k| / Σ_(i∈Ω) V_i ≤ 10⁻³，k=ρ,ρu,ρv,ρE
R_SA / ν̃∞ ≤ 10⁻³
δCL = |CL_now−CL_prev| / max(.1,|CL_now|) ≤ .005
δCD = |CD_now−CD_prev| / max(.01,|CD_now|) ≤ .005
δCM = |CM_now−CM_prev| / max(.01,|CM_now|) ≤ .005</pre>
<p>初次在t*=.01診斷，之後每.5檢查；t*≥3後，上述條件連續5次共同通過才宣稱穩態。不是只有CL/CD拉平。使用者可在convergence設定中明確改門檻，結果保存完整設定，不與原基準混稱相同門檻。</p>
<p>這是近場體積平均raw RHS與載荷穩定條件；不是全域每點殘差、網格獨立性或實驗精度認證。steady未達標回傳退出碼3並保存原因；輸入／數值／輸出失敗回傳2；完成mesh、指定時間或steady達標回傳0。GPU可在batch間檢查wall budget；CPU單次run呼叫內沒有硬式即時中斷。</p>
</section>
<section id="output"><h2>9. 輸出範圍、表面與氣動積分</h2>
<h3>9.1 可選場量與二進位格式</h3><table><tr><th>欄位</th><th>x,y</th><th>p</th><th>rho</th><th>u,v</th><th>mach</th><th>temperature</th></tr><tr><td>單位</td><td>m</td><td>Pa</td><td>kg/m³</td><td>m/s</td><td>1</td><td>K</td></tr></table>
<pre>python3 quick_cfd.py run --points examples/G0001_airfoil_points.csv --fields x y p mach --region -.5 2 -.5 .5 --formats npz
python3 quick_cfd.py export --run runs/my_airfoil --fields x y p rho u v mach temperature --region -.5 2 -.5 .5 --formats npz csv</pre>
<p>region使用x/c、y/c；選取CV重心在框內的格子，值是該控制體平均值，不是規則像素插值。各表包含cell_id與真實fluid_area_m2。NPZ使用未壓縮NumPy格式減少文字轉換；CSV供直接查看。原始<code>internal/*.npy</code>保存完整無因次state和幾何，與使用者選擇的exports分開。</p>
<p><code>output.sample_points_xy_c</code>可指定流體點，輸出nearest fluid CV值及查詢距離；不做插值。點在固體內／壁面或計算域外會拒絕。可用<code>config/field_selection.json</code>作範例。重新export只改產品output檔，保留內部solver資料；不再選取的舊export檔會移除，避免舊CSV與新NPZ混用。</p>
<h3>每案一列的NNLCI氣動表</h3><p><code>aerodynamics.csv/.npz</code>以一列輸出可選載荷欄位。<code>fx/fy</code>為總力；<code>viscous_force_x/y</code>、<code>viscous_drag/lift</code>與<code>viscous_moment_z</code>為黏性分量；pressure分量同理。<code>moment_z</code>為總Mz，<code>moment</code>可作同義選項；CL/CD/CM_z為無因次。完整分量仍保留在loads.json，loads.csv則以pressure／viscous／total三列表示。</p><pre>python3 quick_cfd.py export --run runs/my_airfoil --load-fields drag lift fx fy viscous_force_x viscous_force_y viscous_drag viscous_lift moment_z CL CD CM_z
# JSON：output.load_fields=["drag","lift","fx","fy","moment_z"]
# --no-loads 關閉載荷輸出，--no-surface 關閉壁面表；不影響solver的收斂監測。</pre>
<p>也可只輸出壁面或只輸出氣動係數，減少CSV／NPZ寫檔：</p><pre>python3 quick_cfd.py export --run runs/my_airfoil --no-volume --surface --no-loads
python3 quick_cfd.py export --run runs/my_airfoil --no-volume --no-surface --loads --load-fields CL CD CM_z drag lift</pre><p>JSON對應output.volume／surface／loads三個開關；原始internal仍保留以便日後重新輸出。</p>
<h3>9.2 壁面資料位置與標記</h3><p>surface表逐實際壁段，包含wall_id、source_edge、owner、donor、patch、法向、長度；預設全部patch，可選upper／lower／TE。x/y是壁段中點；<code>p_owner_Pa</code>、<code>rho_owner_kg_m3</code>、<code>temperature_owner_K</code>取鄰壁owner，不能把它們當成完整重建的精確壁面熱力狀態。速度是固定no-slip壁面u=v=0、Mach=0。另輸出Cp、μ_donor、d_n、signed wall shear與各段pressure／viscous force。</p>
<h3>9.3 力、升阻力與Mz</h3><pre class="eq">n_w：流體指向固體
ΔF_pressure = p_w n_w ℓ_w
ΔF_viscous  = −τ_w · n_w ℓ_w
F = Σ_w (ΔF_pressure + ΔF_viscous)
D = Fx cosα + Fy sinα
L = −Fx sinα + Fy cosα
Mz = Σ_w [(x_w−x_ref) ΔFy − (y_w−y_ref) ΔFx]
CD=D/(q∞c), CL=L/(q∞c), CMz=Mz/(q∞c²), q∞=½ρ∞V∞²</pre>
<p><b>xy平面、z軸垂直畫面，Mz逆時針正</b>；這裡不另外把正號改成nose-up。預設參考點(.25c,0)，可用<code>solver.moment_reference_xy_c</code>修改。二維力為每單位展長N/m，力矩N m/m。pressure、viscous、total各自輸出Fx、Fy、drag、lift、Mz及無因次係數；JSON另含每個patch分量。</p>
<p>全翼型載荷總是積分全部原壁段，不隨field ROI或只選上表面而改變。壓力使用絕對靜態p；閉合全翼型常數壓力項幾何上消去，個別patch則不會。signed shear是既有流體應力的切向符號，body viscous force使用其反作用，請以force欄位作積分。</p>
<h3>9.4 目錄與時間定義</h3><pre>run_folder/
  geometry/               原輸入、排序polygon、patch與hash
  config.resolved.json    完整設定
  pipeline.json           整體時間、scale、source/exe hash、各級結果
  index.html              每級摘要入口
  LF1/、LF2/
    settings.txt、command.json、config.resolved.json、solver.log、result.json
    internal/flow.*       完整無因次幾何／state／history／GPU stats
    output/field.npz|csv、surface.npz|csv、loads.json|csv、aerodynamics.npz|csv、mesh.svg、metadata.json</pre>
<p>mesh_setup包含polygon讀取、cut-cell／AMR／合併、幾何檢查及固定table，不含GPU setup／求解。GPU event是trial-batch stream經過時間加總，可能含派發idle，不是kernel busy sum。advance wall含控制與等待，process含該C++程序全部setup／監測／輸出，pipeline另外含點處理、程序啟動和Python export。這些範圍不能全部直接相加或相減。</p>
<p>再次export後，pipeline.json仍記錄原始計算；export.config.json與export.history.json保存目前輸出選項及重新輸出紀錄，不修改求解狀態。</p>
</section>
<section id="editor"><h2>10. 設定檔編輯器</h2><p>下方是此套件的完整預設。可直接改JSON並下載；命令列用<code>--config airfoil_config.json</code>讀取。也可只寫想覆蓋的欄位；未知key會報錯。此頁只編輯設定，不在瀏覽器中執行solver。</p>
<div id="controls"><label>Mach <input id="Mach" type="number" step=".05" value="1.65"></label><label>AOA° <input id="AOA" type="number" step=".5" value="1.5"></label><label>p∞ Pa <input id="pressure" type="number" value="18754"></label><label>T∞ K <input id="temperature" type="number" value="216.65"></label><label>SA <select id="turbulence"><option>sa-neg</option><option>none</option></select></label><label>μ <select id="viscosity"><option>sutherland</option><option>constant</option></select></label><label>CFL <input id="CFL" type="number" step=".01" value=".15"></label><label>wall buffer <input id="buffer" type="number" step="1" value="4"></label><label>LF1 levels <input id="LF1levels" type="number" value="5"></label><label>LF2 levels <input id="LF2levels" type="number" value="6"></label></div>
<p><button id="apply">將控制項套入JSON</button> <button id="download">檢查JSON語法並下載</button> <button id="reset">還原預設</button> <span id="editorStatus"></span></p><textarea id="config" spellcheck="false">__CONFIG_ESCAPED__</textarea>
<p>boxes可在JSON直接編輯。若降低levels，box.level也必須≤該profile最高levels；Python入口會做完整數值檢查，這個下載按鈕只檢查JSON語法。</p>
</section>
<section id="roadmap"><h2>下一次開發：低Mach preconditioning</h2><p><b>使用者於2026-10-20指定：下一階段優先讓整體程式與這個pipeline支援低速流場，熱傳排在後續。</b>本次僅記錄，尚未新增preconditioning實作或參數。</p><p>下次先查文獻、建立M=.1/.01/.001 baseline，評估pseudo-time與低Mach通量／邊界處理，覆蓋cut-cell、黏性、SA與CPU/CUDA；預登記門檻，保留現有LF1/LF2與高速回歸。若做真實非定常，須保留正確physical-time方程。詳見<a href="NEXT_SESSION.md">下次開工記錄</a>。</p><p>熱傳後續再接Fourier／Pr／Prt與定溫／絕熱壁；固體內部熱傳另需能量方程與界面耦合。當前仍κ=0。</p></section>
<section id="validation"><h2>11. 交付時的驗證與已知範圍</h2><p>原核心從已驗證版本獨立複製；<a href="CORE_PROVENANCE.json">CORE_PROVENANCE.json</a>列抽取時hash，新增quick driver只處理參數化與輸出。十翼型LF1/LF2共20個短跑對比封存driver，幾何／五方程／載荷／步數一致；G0001兩級完整steady範例另保存。</p>
<p><a href="validation/ten_short_checks.json">20組比較</a> · <a href="validation/summary.json">完整稽核與實測時間</a> · <a href="validation/controls_checks.json">參數接口檢查</a> · <a href="runs/G0001_default/index.html">完整範例</a>。門檻在實作前登記：場relative L1≤2e-11、Linf≤2e-10、SA欄位relative Linf≤2e-10、步數／時間／拒步相同；SI EOS與壁面積分另核對。</p>
<p class="note">目前目的：快速、穩定、幾乎自動製網的定性aerodynamic資料。LF1/LF2只代表兩套規則，不代表已完成網格獨立性；高Re壁層與湍流阻力尚未認證，κ=0也不支援熱傳預測。低Mach preconditioning、動網格、FSI、3D、動態AMR與implicit仍不在這個pipeline版本中。</p>
</section>
<script>
const defaults=__CONFIG_JSON__,editor=document.getElementById('config'),status=document.getElementById('editorStatus');
document.getElementById('apply').onclick=()=>{try{let c=JSON.parse(editor.value);c.freestream.Mach=+document.getElementById('Mach').value;c.freestream.AOA_degrees=+document.getElementById('AOA').value;c.freestream.p_Pa=+document.getElementById('pressure').value;c.freestream.T_K=+document.getElementById('temperature').value;c.physics.turbulence=document.getElementById('turbulence').value;c.physics.viscosity=document.getElementById('viscosity').value;c.solver.CFL=+document.getElementById('CFL').value;c.solver.time_step={mode:'cfl',dt_seconds:null};c.mesh.wall_buffer=+document.getElementById('buffer').value;for(let key of ['LF1','LF2'])c.mesh.profiles[key].levels=+document.getElementById(key+'levels').value;editor.value=JSON.stringify(c,null,2);status.textContent='已更新JSON，尚未執行計算。'}catch(e){status.textContent=e.message}};
document.getElementById('download').onclick=()=>{try{const c=JSON.parse(editor.value),blob=new Blob([JSON.stringify(c,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='airfoil_config.json';a.click();URL.revokeObjectURL(url);status.textContent='已下載；執行時Python會檢查完整參數。'}catch(e){status.textContent=e.message}};
document.getElementById('reset').onclick=()=>{editor.value=JSON.stringify(defaults,null,2);status.textContent='JSON已還原。'};
</script></html>'''
summary_path=R/'validation/summary.json'
if summary_path.exists():
    summary=json.loads(summary_path.read_text())
    timing='<table><tr><th>G0001 default</th><th>CV</th><th>mesh(s)</th><th>GPU event(s)</th><th>C++ process(s)</th></tr>'+''.join(f'<tr><td>{row["profile"]}</td><td>{row["cells"]}</td><td>{row["mesh_seconds"]:.4f}</td><td>{row["GPU_event_seconds"]:.3f}</td><td>{row["process_seconds"]:.3f}</td></tr>' for row in summary['full_steady'])+'</table>'
    timing+=f'<p>含兩級與Python前後處理的完整pipeline：{summary["default_pipeline_seconds"]:.3f}秒。單一外型、一次實測；不是所有工況的時間保證。兩級均達原定穩態、0reject。</p>'
    body=body.replace('<p><a href="validation/ten_short_checks.json">',timing+'<p><a href="validation/ten_short_checks.json">',1)
body=body.replace('__CONFIG_ESCAPED__',html.escape(json.dumps(c,ensure_ascii=False,indent=2))).replace('__CONFIG_JSON__',json.dumps(c,ensure_ascii=False))
(R/'GUIDE.html').write_text(body)
print(R/'GUIDE.html')
