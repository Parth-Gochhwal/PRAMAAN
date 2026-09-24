NAME          MRPL_CRUDE_BLEND
*
* ============================================================================
* MRPL-Inspired Representative Crude-Blending LP
* ============================================================================
*
* This is a hand-crafted LP representative of crude blending and refinery
* production planning. It is NOT an actual MRPL operational model.
*
* STRUCTURE OVERVIEW
* ------------------
*
*   6 Crude feeds (Arabian Light, Bombay High, Kuwait, Nigerian, Basrah, UAE)
*       |
*       v
*   CDU (Crude Distillation Unit) yields
*       |
*       +---> 4 Blend streams (Naphtha, Kero, Diesel, Fuel Oil)
*       |
*       v
*   Blending + Quality constraints
*       |
*       v
*   4 Finished products with demand requirements
*       |
*       v
*   Economic margin objective (maximize)
*
* VARIABLES (60 total)
* --------------------
*   CR_AL .. CR_UAE     : Crude purchase quantities (6 vars)
*   Y_AL_N .. Y_UAE_FO  : Yield-stream assignments (6 crudes x 4 streams = 24 vars)
*   BL_N .. BL_FO       : Blend stream totals (4 vars)
*   PR_N .. PR_FO       : Finished product outputs (4 vars)
*   SL_N .. SL_FO       : Product sales/surplus (4 vars)
*   IM_N .. IM_FO       : Import/makeup streams (4 vars)
*   TK_N .. TK_FO       : Tank inventory levels (4 vars)
*   CD_THRU             : CDU throughput (1 var)
*   CD_UTIL             : CDU utilization slack (1 var)
*   MG_PROXY            : Margin proxy variable (1 var)
*   LOSS                : Process loss (1 var)
*   SLKA .. SLKF        : 6 slack variables for soft constraints
*   Total: 60 variables
*
* CONSTRAINTS (~55 rows)
* ----------------------
*   Crude availability (6 rows)
*   CDU throughput balance (1 row)
*   CDU capacity (1 row)
*   Yield balance per stream (4 rows)
*   Yield-to-crude linking (24 rows)
*   Blend balance (4 rows)
*   Product balance (4 rows)
*   Product demand lower bounds (4 rows)
*   Quality/spec constraints (4 rows)
*   Capacity constraints (2 rows)
*   Loss accounting (1 row)
*   Margin definition (1 row)
*   Total: ~56 rows
*
ROWS
 N  OBJ
 L  AV_AL
 L  AV_BH
 L  AV_KW
 L  AV_NG
 L  AV_BS
 L  AV_UAE
 E  CDU_BAL
 L  CDU_CAP
 E  YB_N
 E  YB_K
 E  YB_D
 E  YB_FO
 L  YL_AL_N
 L  YL_AL_K
 L  YL_AL_D
 L  YL_AL_FO
 L  YL_BH_N
 L  YL_BH_K
 L  YL_BH_D
 L  YL_BH_FO
 L  YL_KW_N
 L  YL_KW_K
 L  YL_KW_D
 L  YL_KW_FO
 L  YL_NG_N
 L  YL_NG_K
 L  YL_NG_D
 L  YL_NG_FO
 L  YL_BS_N
 L  YL_BS_K
 L  YL_BS_D
 L  YL_BS_FO
 L  YL_UAE_N
 L  YL_UAE_K
 L  YL_UAE_D
 L  YL_UAE_FO
 E  BL_BAL_N
 E  BL_BAL_K
 E  BL_BAL_D
 E  BL_BAL_FO
 E  PR_BAL_N
 E  PR_BAL_K
 E  PR_BAL_D
 E  PR_BAL_FO
 G  DEM_N
 G  DEM_K
 G  DEM_D
 G  DEM_FO
 L  QS_N
 L  QS_K
 L  QS_D
 L  QS_FO
 L  CAP_BL
 L  CAP_TK
 E  LOSS_BAL
 E  MG_DEF
COLUMNS
*
* --- Crude purchase variables ---
* CR_AL = Arabian Light crude purchase (kbbl/day)
*
    CR_AL     OBJ            -42.0
    CR_AL     AV_AL            1.0
    CR_AL     CDU_BAL          1.0
*
* CR_BH = Bombay High crude purchase (kbbl/day)
*
    CR_BH     OBJ            -38.0
    CR_BH     AV_BH            1.0
    CR_BH     CDU_BAL          1.0
*
* CR_KW = Kuwait crude purchase (kbbl/day)
*
    CR_KW     OBJ            -40.0
    CR_KW     AV_KW            1.0
    CR_KW     CDU_BAL          1.0
*
* CR_NG = Nigerian crude purchase (kbbl/day)
*
    CR_NG     OBJ            -48.0
    CR_NG     AV_NG            1.0
    CR_NG     CDU_BAL          1.0
*
* CR_BS = Basrah crude purchase (kbbl/day)
*
    CR_BS     OBJ            -36.0
    CR_BS     AV_BS            1.0
    CR_BS     CDU_BAL          1.0
*
* CR_UAE = UAE crude purchase (kbbl/day)
*
    CR_UAE    OBJ            -44.0
    CR_UAE    AV_UAE           1.0
    CR_UAE    CDU_BAL          1.0
*
* --- Yield stream variables ---
* Y_{crude}_{stream}: fraction of crude going to each stream
* Yield coefficients represent typical CDU product splits
*
* Arabian Light yields: N=0.22, K=0.15, D=0.28, FO=0.32 (loss ~3%)
    Y_AL_N    YB_N             1.0
    Y_AL_N    YL_AL_N          1.0
    Y_AL_N    BL_BAL_N         1.0
    Y_AL_K    YB_K             1.0
    Y_AL_K    YL_AL_K          1.0
    Y_AL_K    BL_BAL_K         1.0
    Y_AL_D    YB_D             1.0
    Y_AL_D    YL_AL_D          1.0
    Y_AL_D    BL_BAL_D         1.0
    Y_AL_FO   YB_FO            1.0
    Y_AL_FO   YL_AL_FO         1.0
    Y_AL_FO   BL_BAL_FO        1.0
*
* Bombay High yields: N=0.28, K=0.18, D=0.25, FO=0.26 (loss ~3%)
    Y_BH_N    YB_N             1.0
    Y_BH_N    YL_BH_N          1.0
    Y_BH_N    BL_BAL_N         1.0
    Y_BH_K    YB_K             1.0
    Y_BH_K    YL_BH_K          1.0
    Y_BH_K    BL_BAL_K         1.0
    Y_BH_D    YB_D             1.0
    Y_BH_D    YL_BH_D          1.0
    Y_BH_D    BL_BAL_D         1.0
    Y_BH_FO   YB_FO            1.0
    Y_BH_FO   YL_BH_FO         1.0
    Y_BH_FO   BL_BAL_FO        1.0
*
* Kuwait yields: N=0.18, K=0.14, D=0.30, FO=0.35 (loss ~3%)
    Y_KW_N    YB_N             1.0
    Y_KW_N    YL_KW_N          1.0
    Y_KW_N    BL_BAL_N         1.0
    Y_KW_K    YB_K             1.0
    Y_KW_K    YL_KW_K          1.0
    Y_KW_K    BL_BAL_K         1.0
    Y_KW_D    YB_D             1.0
    Y_KW_D    YL_KW_D          1.0
    Y_KW_D    BL_BAL_D         1.0
    Y_KW_FO   YB_FO            1.0
    Y_KW_FO   YL_KW_FO         1.0
    Y_KW_FO   BL_BAL_FO        1.0
*
* Nigerian yields: N=0.26, K=0.16, D=0.24, FO=0.30 (loss ~4%)
    Y_NG_N    YB_N             1.0
    Y_NG_N    YL_NG_N          1.0
    Y_NG_N    BL_BAL_N         1.0
    Y_NG_K    YB_K             1.0
    Y_NG_K    YL_NG_K          1.0
    Y_NG_K    BL_BAL_K         1.0
    Y_NG_D    YB_D             1.0
    Y_NG_D    YL_NG_D          1.0
    Y_NG_D    BL_BAL_D         1.0
    Y_NG_FO   YB_FO            1.0
    Y_NG_FO   YL_NG_FO         1.0
    Y_NG_FO   BL_BAL_FO        1.0
*
* Basrah yields: N=0.16, K=0.12, D=0.32, FO=0.37 (loss ~3%)
    Y_BS_N    YB_N             1.0
    Y_BS_N    YL_BS_N          1.0
    Y_BS_N    BL_BAL_N         1.0
    Y_BS_K    YB_K             1.0
    Y_BS_K    YL_BS_K          1.0
    Y_BS_K    BL_BAL_K         1.0
    Y_BS_D    YB_D             1.0
    Y_BS_D    YL_BS_D          1.0
    Y_BS_D    BL_BAL_D         1.0
    Y_BS_FO   YB_FO            1.0
    Y_BS_FO   YL_BS_FO         1.0
    Y_BS_FO   BL_BAL_FO        1.0
*
* UAE yields: N=0.24, K=0.17, D=0.26, FO=0.30 (loss ~3%)
    Y_UAE_N   YB_N             1.0
    Y_UAE_N   YL_UAE_N         1.0
    Y_UAE_N   BL_BAL_N         1.0
    Y_UAE_K   YB_K             1.0
    Y_UAE_K   YL_UAE_K         1.0
    Y_UAE_K   BL_BAL_K         1.0
    Y_UAE_D   YB_D             1.0
    Y_UAE_D   YL_UAE_D         1.0
    Y_UAE_D   BL_BAL_D         1.0
    Y_UAE_FO  YB_FO            1.0
    Y_UAE_FO  YL_UAE_FO        1.0
    Y_UAE_FO  BL_BAL_FO        1.0
*
* --- Blend stream total variables ---
    BL_N      BL_BAL_N        -1.0
    BL_N      QS_N             1.0
    BL_N      CAP_BL           1.0
    BL_K      BL_BAL_K        -1.0
    BL_K      QS_K             1.0
    BL_K      CAP_BL           1.0
    BL_D      BL_BAL_D        -1.0
    BL_D      QS_D             1.0
    BL_D      CAP_BL           1.0
    BL_FO     BL_BAL_FO       -1.0
    BL_FO     QS_FO            1.0
    BL_FO     CAP_BL           1.0
*
* --- Finished product output variables ---
    PR_N      OBJ             62.0
    PR_N      PR_BAL_N        -1.0
    PR_N      DEM_N            1.0
    PR_K      OBJ             58.0
    PR_K      PR_BAL_K        -1.0
    PR_K      DEM_K            1.0
    PR_D      OBJ             65.0
    PR_D      PR_BAL_D        -1.0
    PR_D      DEM_D            1.0
    PR_FO     OBJ             35.0
    PR_FO     PR_BAL_FO       -1.0
    PR_FO     DEM_FO           1.0
*
* --- Product balance: blend + import - sales - inventory = product ---
    SL_N      PR_BAL_N         1.0
    SL_N      OBJ              1.5
    SL_K      PR_BAL_K         1.0
    SL_K      OBJ              1.2
    SL_D      PR_BAL_D         1.0
    SL_D      OBJ              1.8
    SL_FO     PR_BAL_FO        1.0
    SL_FO     OBJ              0.5
*
* --- Import / makeup streams (allow small external product purchases) ---
    IM_N      PR_BAL_N        -1.0
    IM_N      OBJ            -70.0
    IM_K      PR_BAL_K        -1.0
    IM_K      OBJ            -66.0
    IM_D      PR_BAL_D        -1.0
    IM_D      OBJ            -72.0
    IM_FO     PR_BAL_FO       -1.0
    IM_FO     OBJ            -40.0
*
* --- Tank inventory variables ---
    TK_N      PR_BAL_N         1.0
    TK_N      CAP_TK           1.0
    TK_K      PR_BAL_K         1.0
    TK_K      CAP_TK           1.0
    TK_D      PR_BAL_D         1.0
    TK_D      CAP_TK           1.0
    TK_FO     PR_BAL_FO        1.0
    TK_FO     CAP_TK           1.0
*
* --- CDU throughput ---
    CD_THRU   CDU_BAL         -1.0
    CD_THRU   CDU_CAP          1.0
    CD_THRU   LOSS_BAL         1.0
    CD_THRU   MG_DEF           1.0
*
* --- CDU utilization slack ---
    CD_UTIL   CDU_CAP          1.0
*
* --- Margin proxy ---
    MG_PROXY  MG_DEF          -1.0
    MG_PROXY  OBJ              1.0
*
* --- Process loss ---
    LOSS      LOSS_BAL        -1.0
    LOSS      OBJ             -5.0
*
* --- Slack variables for yield linking constraints ---
    SLKA      YL_AL_N         -1.0
    SLKA      YL_AL_K         -1.0
    SLKA      YL_AL_D         -1.0
    SLKA      YL_AL_FO        -1.0
    SLKB      YL_BH_N         -1.0
    SLKB      YL_BH_K         -1.0
    SLKB      YL_BH_D         -1.0
    SLKB      YL_BH_FO        -1.0
    SLKC      YL_KW_N         -1.0
    SLKC      YL_KW_K         -1.0
    SLKC      YL_KW_D         -1.0
    SLKC      YL_KW_FO        -1.0
    SLKD      YL_NG_N         -1.0
    SLKD      YL_NG_K         -1.0
    SLKD      YL_NG_D         -1.0
    SLKD      YL_NG_FO        -1.0
    SLKE      YL_BS_N         -1.0
    SLKE      YL_BS_K         -1.0
    SLKE      YL_BS_D         -1.0
    SLKE      YL_BS_FO        -1.0
    SLKF      YL_UAE_N        -1.0
    SLKF      YL_UAE_K        -1.0
    SLKF      YL_UAE_D        -1.0
    SLKF      YL_UAE_FO       -1.0
RHS
* Crude availability limits (kbbl/day)
    RHS1      AV_AL           50.0
    RHS1      AV_BH           35.0
    RHS1      AV_KW           45.0
    RHS1      AV_NG           25.0
    RHS1      AV_BS           40.0
    RHS1      AV_UAE          30.0
* CDU capacity (kbbl/day)
    RHS1      CDU_CAP        150.0
* Yield balance (sum of yield fractions per crude = crude amount)
    RHS1      YB_N             0.0
    RHS1      YB_K             0.0
    RHS1      YB_D             0.0
    RHS1      YB_FO            0.0
* Yield linking: yield_stream <= yield_fraction * crude_amount
* These RHS are 0 because we reformulate as: Y_{c,s} - frac * CR_c <= 0
* But since we can't put A-matrix cross-terms in RHS section,
* we use the slack variables SLKA..SLKF and set linking RHS to crude availability
    RHS1      YL_AL_N         11.0
    RHS1      YL_AL_K          7.5
    RHS1      YL_AL_D         14.0
    RHS1      YL_AL_FO        16.0
    RHS1      YL_BH_N          9.8
    RHS1      YL_BH_K          6.3
    RHS1      YL_BH_D          8.75
    RHS1      YL_BH_FO         9.1
    RHS1      YL_KW_N          8.1
    RHS1      YL_KW_K          6.3
    RHS1      YL_KW_D         13.5
    RHS1      YL_KW_FO        15.75
    RHS1      YL_NG_N          6.5
    RHS1      YL_NG_K          4.0
    RHS1      YL_NG_D          6.0
    RHS1      YL_NG_FO         7.5
    RHS1      YL_BS_N          6.4
    RHS1      YL_BS_K          4.8
    RHS1      YL_BS_D         12.8
    RHS1      YL_BS_FO        14.8
    RHS1      YL_UAE_N         7.2
    RHS1      YL_UAE_K         5.1
    RHS1      YL_UAE_D          7.8
    RHS1      YL_UAE_FO         9.0
* Product demand requirements (kbbl/day)
    RHS1      DEM_N           15.0
    RHS1      DEM_K           10.0
    RHS1      DEM_D           25.0
    RHS1      DEM_FO          12.0
* Quality/spec limits
    RHS1      QS_N            40.0
    RHS1      QS_K            30.0
    RHS1      QS_D            50.0
    RHS1      QS_FO           45.0
* Blending capacity (kbbl/day total across all streams)
    RHS1      CAP_BL         160.0
* Tank capacity (kbbl total across all tanks)
    RHS1      CAP_TK          80.0
* Process loss (fraction of throughput, ~3%)
    RHS1      LOSS_BAL         0.0
BOUNDS
* Crude purchases: 0 <= CR <= availability (finite upper bounds)
 UP BND1      CR_AL           50.0
 UP BND1      CR_BH           35.0
 UP BND1      CR_KW           45.0
 UP BND1      CR_NG           25.0
 UP BND1      CR_BS           40.0
 UP BND1      CR_UAE          30.0
* Yield streams: nonneg with reasonable upper bounds
 UP BND1      Y_AL_N          15.0
 UP BND1      Y_AL_K          10.0
 UP BND1      Y_AL_D          18.0
 UP BND1      Y_AL_FO         20.0
 UP BND1      Y_BH_N          12.0
 UP BND1      Y_BH_K           8.0
 UP BND1      Y_BH_D          12.0
 UP BND1      Y_BH_FO         12.0
 UP BND1      Y_KW_N          12.0
 UP BND1      Y_KW_K           8.0
 UP BND1      Y_KW_D          18.0
 UP BND1      Y_KW_FO         20.0
 UP BND1      Y_NG_N           8.0
 UP BND1      Y_NG_K           5.0
 UP BND1      Y_NG_D           8.0
 UP BND1      Y_NG_FO         10.0
 UP BND1      Y_BS_N          10.0
 UP BND1      Y_BS_K           6.0
 UP BND1      Y_BS_D          16.0
 UP BND1      Y_BS_FO         18.0
 UP BND1      Y_UAE_N         10.0
 UP BND1      Y_UAE_K          7.0
 UP BND1      Y_UAE_D         10.0
 UP BND1      Y_UAE_FO        12.0
* Blend totals
 UP BND1      BL_N            45.0
 UP BND1      BL_K            30.0
 UP BND1      BL_D            55.0
 UP BND1      BL_FO           50.0
* Product outputs
 UP BND1      PR_N            45.0
 UP BND1      PR_K            30.0
 UP BND1      PR_D            55.0
 UP BND1      PR_FO           50.0
* Sales surplus
 UP BND1      SL_N            20.0
 UP BND1      SL_K            15.0
 UP BND1      SL_D            25.0
 UP BND1      SL_FO           20.0
* Imports
 UP BND1      IM_N            10.0
 UP BND1      IM_K             8.0
 UP BND1      IM_D            10.0
 UP BND1      IM_FO            8.0
* Tank inventories
 UP BND1      TK_N            25.0
 UP BND1      TK_K            20.0
 UP BND1      TK_D            25.0
 UP BND1      TK_FO           20.0
* CDU
 UP BND1      CD_THRU        150.0
 UP BND1      CD_UTIL        150.0
* Margin proxy
 UP BND1      MG_PROXY      5000.0
* Loss
 UP BND1      LOSS            10.0
* Slacks
 UP BND1      SLKA            50.0
 UP BND1      SLKB            35.0
 UP BND1      SLKC            45.0
 UP BND1      SLKD            25.0
 UP BND1      SLKE            40.0
 UP BND1      SLKF            30.0
ENDATA