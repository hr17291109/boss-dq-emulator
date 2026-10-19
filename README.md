# powerspectrum

Dark Quest Simulation のハローカタログと SDSS/BOSS CMASS 銀河の赤方偏移空間パワースペクトル多重極を直接比較し、宇宙論パラメータを推定するためのコード。

## Overview

摂動論に基づく解析では使える波数が線形〜準線形領域に限られる。このリポジトリでは、N 体シミュレーションから作った銀河カタログのパワースペクトルを観測と直接比較することで、非線形スケールまで情報を引き出すことを目指す。

推論は二段構えになっている（適合度エミュレータ）。

1. **内側（C++）**：各宇宙論（Dark Quest の 1 シミュレーション）ごとに、銀河-ハロー関係のパラメータを Metropolis 法で探索し、観測との $\chi^2$ を記録する。
2. **外側（Python）**：宇宙論ごとに得た適合度（スカラー量）を宇宙論パラメータの関数としてガウス過程回帰し、それを尤度として宇宙論パラメータの事後分布を MCMC で求める。

出力が 1 次元のスカラーなので、$P(k)$ そのものを回帰する多次元エミュレータに比べて精度検証と誤差伝播の評価がしやすい。

> 適合度として最小 $\chi^2$ と周辺化尤度のどちらを使うかは検討中。

### 銀河-ハロー関係

1 ハロー = 1 銀河とし、$V_\mathrm{max}$ の滑らかな閾値で選ぶ（HOD ではない）。

$$
P_\mathrm{sel}(V_\mathrm{max}) = \frac{1}{2}\left[1 + \tanh\left(\frac{V_\mathrm{max} - V_\mathrm{max,th}}{\Delta V_\mathrm{max}}\right)\right]
$$

自由パラメータは $V_\mathrm{max,th}$（閾値）と $\Delta V_\mathrm{max}$（遷移幅）の 2 つ。赤方偏移空間への写像にはハローの**コア速度**を使う。

## Pipeline

```
[C++]    halos_mcmc.exe configs/<config>.yaml
           ハロー選択 (tanh 確率) → 3 視線方向で P_ℓ(k) 測定 (CIC + interlacing, AP 込み)
           → 窓関数畳み込み → χ² → Metropolis で (ΔV_max, V_max,th) を探索
           ↓  output/Q*/..._cov_chi2.dat
[Python] 宇宙論ごとの適合度を集約 → GP 回帰 → 宇宙論 MCMC → 事後分布
```

別系統として、Lange et al. (2019, CEM) に倣ったベイズエビデンスの計算がある。

## Status

| 構成要素 | ファイル | 場所 |
|---|---|---|
| P(k) 測定と銀河-ハロー関係 MCMC | `halos_mcmc.cpp` ほか C++ ヘッダ | `main` |
| 観測データとモデルの比較・χ² | `pk_compare.py`, `pk_tools.py`, `scpt.py` | `main` |
| 内側 MCMC の診断・提案共分散の調整 | `plot_ghc_mcmc.py`, `run_mcmc_diagnostics.py` | `main` |
| 適合度エミュレータ（GP + 宇宙論 MCMC） | `cosmo_inference.py` | ブランチ `fix/cosmo-inference-mcmc` |
| エビデンス計算 | `cem/`, `run_evidence.py` | ブランチ `evidence-cem` |

## Requirements

### C++

- GCC（g++）<!-- TODO: compile.sh の -std= に合わせて C++ 標準を書く -->
- [Eigen 3](https://eigen.tuxfamily.org/)
- [GSL](https://www.gnu.org/software/gsl/)（乱数、スプライン補間）
- [FFTW 3](http://www.fftw.org/)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)（設定ファイル）
- OpenMP（密度場の割り当てと FFT の並列化）

### Python

- Python 3
- numpy, scipy, pandas, matplotlib
- scikit-learn（ガウス過程回帰）
- [GetDist](https://getdist.readthedocs.io/)（等高線図、Gelman-Rubin 統計量）
- tqdm
- （任意）jax, blackjax

## Build

```bash
git clone git@github.com:hr17291109/powerspectrum.git
# または
git clone https://github.com/hr17291109/powerspectrum.git

cd powerspectrum
sh compile.sh
```

`compile.sh` は `git describe --always --dirty` の結果を `GIT_HASH` マクロとして埋め込む。赤方偏移空間での測定には `-DZSPACE` が必要。`-DUSE_BULKVEL` は付けない（コア速度を使う）。

<!-- TODO: Eigen / GSL / FFTW / yaml-cpp のパスを環境ごとにどう指定するか -->

## Usage

### 1. 銀河-ハロー関係の MCMC（C++）

```bash
cp configs_example/<template>.yaml configs/Q1_highz_ngc.yaml   # 編集して使う
./halos_mcmc.exe configs/Q1_highz_ngc.yaml
```

`configs/` はチェーンごとに書き換えるため Git では追跡していない。ひな形は `configs_example/` にある。

| キー | 意味 |
|---|---|
| `file_base` | ハローカタログのパス |
| `snapnum` | スナップショット番号（赤方偏移） |
| `output_base` | 出力ファイル名の接頭辞 |
| `v_th`, `delta_v` | 初期値 $V_\mathrm{max,th}$, $\Delta V_\mathrm{max}$ |
| `ns_type` | 0 = NGC, 1 = SGC |
| `n_mc` | MCMC のステップ数 |
| `kmax` | **フィット範囲の上限**（モデルの $k$ グリッドではない。下記 Conventions） |
| `mcmc_seed` | MCMC の乱数シード（省略時 1） |
| `step_covariance` | 提案分布の 2×2 共分散（**必須**） |

出力：

- `output/Q*/<output_base>_cov_chi2.dat` — 採択されたチェーン
- `output/Q*/<output_base>_others_cov_chi2.dat` — 棄却された提案（採択率の計算用）

### 2. チェーンの診断

```bash
python run_mcmc_diagnostics.py
```

採択率、提案共分散の推奨値（`format_step_covariance` の出力をそのまま YAML に貼れる形式）、次のチェーンの開始点、等高線図、$\hat R$ を出す。採択率 0.3〜0.4 を目安に、短いチェーン → 診断 → 共分散の更新を繰り返してから本番を走らせる。

> $\hat R$ の規約：GetDist の `getGelmanRubin()` は $R-1$、blackjax の `potential_scale_reduction` は $R$ そのものを返す。

### 3. 観測データとの比較

```bash
python scpt.py
```

`pk_compare.py` が BOSS の多重極・窓関数行列・共分散を読み、シミュレーションの $P_\ell(k)$ を畳み込んで $\chi^2$ と比較図を出す。

<!-- TODO: 入力データ（ハローカタログ、BOSS の P(k)、窓関数行列、共分散）の置き場所と入手方法 -->

## Data

| 項目 | 内容 |
|---|---|
| 観測 | BOSS CMASS、NGC / SGC × LOWZ ($z=0.38$) / HIGHZ ($z=0.61$)、$\ell = 0, 2, 4$ |
| 共分散 | 2048 個のモックから推定 |
| シミュレーション | Dark Quest、30 宇宙論（`Q0000`〜）、箱 1000 $h^{-1}$Mpc |
| 宇宙論パラメータ（GP 入力） | $\Omega_m$, $w_0$, $A_s$, $n_s$ |

## Conventions

数値に直接効く取り決め。変更する場合はここも更新すること。

**$k$ グリッド**
- モデル側のグリッドは窓関数行列のサイズで固定されており、`binning.hpp` の `constexpr`（`nbins = 400`, `kmin = 0.0`, `kmax = 0.4`, 線形ビン）で定義している。YAML から変えてはいけない。
  - `M`（2000×1200）：3 多重極 × 400 本（$\Delta k = 0.001$）を入力
  - `W`（200×2000）：5 多重極 × 40 本（$\Delta k = 0.01$）を出力
- 窓関数の畳み込みは $k$ を混ぜるため、低 $k$ だけフィットする場合でもモデルは $k = 0.4$ まで計算する。
- YAML の `kmax` はフィット範囲の上限のみ。使うビン数は `limit = round(kmax / 0.01)`（1〜40）、データ点数は `3 × limit`。

**P(k) の測定**
- 格子 $512^3$、CIC 割り当て + interlacing。
- 視線方向は $x, y, z$ の 3 方向で測定する。
- AP 効果：基準宇宙論 $\Omega_{m,\mathrm{fid}} = 0.31$ の座標に写して測る。$H(z)$ は各シミュレーションの展開表から補間する。
- シミュレーション側のショットノイズはモノポールから差し引いている。

**統計**
- 共分散の逆行列には Hartlap 補正 $\alpha = (n-p-2)/(n-1)$（$n = 2048$, $p$ = フィットに使うデータ点数）を掛ける。
  - Python（`pk_compare.py`）：常に適用。
  - C++：`chi_square_hartlap` を使う場合に適用。**現在はソースで切り替えている**ので、出力がどちらで作られたかを別途記録すること。
- 逆行列は陽に作らず Cholesky 分解（C++ は LDLT）で解く。

**再現性**
- ハロー選択の乱数シードは `12345` で固定（全ステップ共通）。
- 同じチェーンをビット単位で再現したい場合は `OMP_NUM_THREADS=1` で実行する。多スレッドでは加算順序の違いで相対 $10^{-6}$ 程度ずれる。
- ハロー選択ループは共有 RNG を使うので並列化しないこと。

## Known limitations

- フィット範囲の下限（`fit_kmin`）は未実装。
- $k < k_f \simeq 0.0063$ の細かいビンにはモードが無く、0 として畳み込みに入る。
- C++ の出力ファイルには実行条件（コミット、設定、Hartlap の有無）のヘッダがまだ無い。

## References

- Lange et al. 2019, arXiv:1909.03107 — Cosmological Evidence Modelling
- Beutler et al. — BOSS パワースペクトル多重極と `pk_tools.py` の元コード
- Hartlap et al. 2007 — 逆共分散の補正
