# 下次開工記錄

## 下一階段優先：低 Mach preconditioning（使用者指定，2026-10-20）

使用者要求本次先記錄，下一次優先讓整個程式與獨立airfoil pipeline支援低速流場。
順序調整為：**低Mach preconditioning → 後續熱傳**。本次只更新紀錄，尚未實作preconditioner。

1. 下次先查paper/book並盤點現有density-based離散，建立M=0.1／0.01／0.001的低速baseline，記清物理參數與Re。
2. 評估時間／pseudo-time預條件與低Mach通量耗散的一致處理，並納入farfield／outlet、cut-cell、黏性、SA及GPU路徑。
3. 穩態加速與真實非定常分開設計；若採dual-time，保留正確physical-time導數，不直接改變真實時間物理。
4. 事前登記壓力／速度誤差、守恆、正性、穩態與CPU/GPU門檻；沿用LF1/LF2與十翼型回歸，保留跨音／超音速既有能力。
5. 量測可用解總時間、步數、GPU event／advance／process，不只看每步或GPU使用率；不預先承諾倍率。

熱傳後續再接：流體Fourier熱通量、Pr／Prt、定溫／絕熱壁；若要固體內部溫度，再加固體能量方程及流固熱耦合。
目前仍κ=0；選定固體材料本身不代表已完成熱傳。現有可輸出溫度是由氣體總能量／EOS恢復。

當前入口：quick_cfd.py／config/default.json／GUIDE.html。LF1約4700CV、LF2約9500CV；SA-neg＋Sutherland＋fused，M1.65既有回歸保留。
