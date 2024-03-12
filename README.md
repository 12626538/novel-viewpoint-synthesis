# novel-viewpoint-synthesis

## Results
These are currently the best results. These metrics are subject to constant change.

Results are after 7.000 and 30.000 training iterations, unless otherwise specified.

* **mipnerf/bicycle**

    <!-- model: mar12tue114913 -->
    | Other                 | PSNR  | LPIPS | SSIM |
    |-----------------------|-------|-------|------|
    | **Best so far** - 7K  | 22.28 | 0.49  | 0.59 |
    | Inria-7K              | 24.11 | 0.31  | 0.69 |
    | Splatfacto-7K         | 22.99 | 0.31  | 0.65 |
    | Splatfacto-7K (big)   | 23.66 | 0.28  | 0.69 |
    | **Best so far** - 30K | 23.81 | 0.36  | 0.67 |
    | Inria-30K             | 25.61 | 0.21  | 0.78 |
    | Splatfacto-30K        | 24.99 | 0.18  | 0.75 |
    | Splatfacto-30K (big)  | 25.7  | 0.15  | 0.78 |
    [source](https://docs.gsplat.studio/tests/eval.html)

* **mipnerf/bonsai**
    <!-- model: mar12tue144510 -->

    | Other                 | PSNR  | LPIPS | SSIM |
    |-----------------------|-------|-------|------|
    | **Best so far** - 7K  | 27.93 | 0.20  | 0.90 |
    | Inria-7K              | 29.49 | 0.24  | 0.92 |
    | Splatfacto-7K         | 29.45 | 0.16  | 0.92 |
    | Splatfacto-7K (big)   | 29.69 | 0.16  | 0.92 |
    | **Best so far** - 30K | 30.99 | 0.16  | 0.93 |
    | Inria-30K             | 31.89 | 0.21  | 0.94 |
    | Splatfacto-30K        | 32.14 | 0.13  | 0.94 |
    | Splatfacto-30K (big)  | 32.23 | 0.13  | 0.94 |
    [source](https://docs.gsplat.studio/tests/eval.html)

* **3DU/seem_local_house_269**
    <!-- mar12tue140040 -->
    | Metric      |  7K iter  | 30K iter  |
    |-------------|-----------|-----------|
    | **DSSIM**   |  0.134    |  0.101    |
    | **SSIM**    |  0.866    |  0.899    |
    | **PSNR**    | 24.6      | 27.19     |
    | **LPIPS**   |  0.276    |  0.203    |
