import numpy as np
from enum import Enum
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import mplcursors to hove over a curve in a plot to display values.Doesn't seem to work with 3 axis plots

# Plot Kp/v(j) as funcion of joint positions

# The leg joints are as follows:
# 1) Translation 2:2, 1 motor: Crotch P/R and Ankle P/R
# 2) Rotation, 1:1, 1 motor: Crotch Y
# 3) Translation, 1:1, 2 motors: Knee P

# The joint gains used in mujoco simulations are as follows:
#         pdgains.T[0] = np.array([100, 100, 100, 125, 40, 40,
#                                          100, 100, 100, 125, 40, 40])
#         pdgains.T[1] = np.array([10, 10, 10, 12.5, 4, 4,
#                                          10, 10, 10, 12.5, 4, 4])
# I will start by choosing motor gains that give joint gains as above in half sitting posture 
# Then, I will iterate over the joint positions in the joint range and see how the joint gains change
# Half Sitting posture: [crotch_Y, crotch_P, crotch_R, knee_P, ankle_P, ankle_R] = [0, 0, -17, 36, 0, -19]

# Define a gimbal method as enum

class GimbalMethod(Enum):
    PITCH_ROLL = 0
    ROLL_PITCH = 1

# Define a function that converts a angle (joint) to cylinder position

def kr001_angle_to_cylinder_gimbal(GimbalMethod, qr, qp, LT, LC, L1Y, L1Z, L2X, L2Y, L2Z, L3X, L3Z, LTxLT,LCxLC, LCx2, QOFF2):

    cqp = math.cos(qp)
    sqp = math.sin(qp)
    cqr = math.cos(qr)
    sqr = math.cos(qr)

    cqpL2X = cqp * L2X
    sqpL2X = sqp * L2X
    cqrL2Y = cqr * L2Y
    sqrL2Y = sqr * L2Y
    cqrL2Z = cqr * L2Z
    sqrL2Z = sqr * L2Z
    cqpL2Z = cqp * L2Z
    sqpL2Z = sqp * L2Z

    # Left Cylinder

    if GimbalMethod.value == 1: #ROLL_PITCH
        # |  cos(qp)  0   sin(qp) |   |  1,        0,        0 |   | -L2X |
        # |        0  1        0  | * |  0,  cos(qr), -sin(qr) | * | -L2Y |
        # | -sin(qp)  0   cos(qp) |   |  0,  sin(qr),  cos(qr) |   |  L2Z |
       cqrL2Z_sqrL2Y = (cqrL2Z - sqrL2Y) 
       phx = -cqpL2X + sqp * cqrL2Z_sqrL2Y
       phy = -cqrL2Y - sqrL2Z
       phz =  sqpL2X + cqp * cqrL2Z_sqrL2Y
    elif GimbalMethod.value == 0: #PITCH_ROLL
        # |  1,        0,        0 |   |  cos(qp)  0   sin(qp) |   | -L2X |
        # |  0,  cos(qr), -sin(qr) | * |        0  1        0  | * | -L2Y |
        # |  0,  sin(qr),  cos(qr) |   | -sin(qp)  0   cos(qp) |   |  L2Z |      
        cqpL2Z_sqpL2X = (cqpL2Z + sqpL2X)
        phx = -cqpL2X + sqpL2Z
        phy = -cqrL2Y - sqr * cqpL2Z_sqpL2X
        phz = -sqrL2Y + cqr * cqpL2Z_sqpL2X
    else:
        print("Invalid Gimbal Method Type! ")
        return 1
    phzoffs = (L1Z + phz)
    phyoffs = (L1Y + phy)

    phx2phz2   = (phx * phx) + (phzoffs * phzoffs)
    phx2phz2sq = math.sqrt(phx2phz2)

    sqcqv = (LTxLT - (phyoffs * phyoffs) - LCxLC - phx2phz2) / (LCx2 * phx2phz2sq)
    
    qoffs = math.atan2(phx, phzoffs)
    qcL   = math.acos(sqcqv) + qoffs + QOFF2

    cqc = math.cos(qcL)
    sqc = math.sin(qcL)

    dax = (LC * sqc - L3X)
    daz = (LC * cqc - L3Z)

    daLdb = (dax * dax) + (daz * daz)
    leftLength = math.sqrt(daLdb)

    # Right Cylinder
    if GimbalMethod.value == 1: #ROLL_PITCH
        # |  cos(qp)  0   sin(qp) |   |  1,        0,        0 |   | -L2X |
        # |        0  1        0  | * |  0,  cos(qr), -sin(qr) | * | +L2Y |
        # | -sin(qp)  0   cos(qp) |   |  0,  sin(qr),  cos(qr) |   |  L2Z |
       cqrL2Z_sqrL2Y = (cqrL2Z + sqrL2Y) 
       phx = -cqpL2X + sqp * cqrL2Z_sqrL2Y
       phy = cqrL2Y - sqrL2Z
       phz =  sqpL2X + cqp * cqrL2Z_sqrL2Y
    elif GimbalMethod.value == 0: #PITCH_ROLL
        # |  1,        0,        0 |   |  cos(qp)  0   sin(qp) |   | -L2X |
        # |  0,  cos(qr), -sin(qr) | * |        0  1        0  | * | -L2Y |
        # |  0,  sin(qr),  cos(qr) |   | -sin(qp)  0   cos(qp) |   |  L2Z |      
        cqpL2Z_sqpL2X = (cqpL2Z + sqpL2X)
        phx = -cqpL2X + sqpL2Z
        phy = cqrL2Y - sqr * cqpL2Z_sqpL2X
        phz = sqrL2Y + cqr * cqpL2Z_sqpL2X
    else:
        print("Invalid Gimbal Method Type! ")
        return 1
    
    phzoffs = (L1Z + phz)
    phyoffs = (L1Y - phy)

    phx2phz2   = (phx * phx) + (phzoffs * phzoffs)
    phx2phz2sq = math.sqrt(phx2phz2)

    sqcqv = (LTxLT - (phyoffs * phyoffs) - LCxLC - phx2phz2) / (LCx2 * phx2phz2sq)

    qoffs = math.atan2(phx, phzoffs)
    qcR   = math.acos(sqcqv) + qoffs + QOFF2

    cqc = math.cos(qcR)
    sqc = math.sin(qcR)

    dax = (LC * sqc - L3X)
    daz = (LC * cqc - L3Z)

    daRdb = (dax * dax) + (daz * daz)
    rightLength   = math.sqrt(daRdb)

    return leftLength, rightLength

# Define a function that gives the inverse Jacobian:
# From line 379 in  r001_kinematics.cpp:
# 静力学による近似 F = -invJ * T 
# In our calculations, we have: Fc = TransJj * Tj
# We need Jj inverted, with and without transpose for the final calculations, which means it's enough to take -J from these files, as it corresponds to invJj

def Jacobian_MotorTorque_CylinderForce(GimbalMethod, THR, THP, LT, LC, L1Y, L1Z, L2X, L2Y, L2Z, L3X, L3Z, LTxLT,LCxLC, LCx2, QOFF2):
    
    dth = math.radians(0.01)

    # Initialize variables
    drL = 0.0
    drR = 0.0
    dpL = 0.0
    dpR = 0.0

    # Current Lengths
    drL, drR = kr001_angle_to_cylinder_gimbal(GimbalMethod, THR, THP, LT, LC, L1Y, L1Z, L2X, L2Y, L2Z, L3X, L3Z, LTxLT, LCxLC, LCx2, QOFF2)

    curLength = np.array([drL, drR]) 

    # Roll: Call the function again with THR + dth
    drL, drR = kr001_angle_to_cylinder_gimbal(GimbalMethod, THR + dth, THP, LT, LC, L1Y, L1Z, L2X, L2Y, L2Z, L3X, L3Z, LTxLT, LCxLC, LCx2, QOFF2)

    drLength = np.array([drL, drR])
    drLength = (drLength - curLength) / dth  # Equivalent to vector subtraction and division

    # Pitch: Call the function with THP + dth
    dpL, dpR = kr001_angle_to_cylinder_gimbal(GimbalMethod, THR, THP + dth, LT, LC, L1Y, L1Z, L2X, L2Y, L2Z, L3X, L3Z, LTxLT, LCxLC, LCx2, QOFF2)

    dpLength = np.array([dpL, dpR])
    dpLength = (dpLength - curLength) / dth 

    jacobi = np.array([[drLength[0], dpLength[0]], [drLength[1], dpLength[1]]])
    
    return -jacobi

# This function is required for "Knee cylinder to angle" just below it:
def circleIntersectPoint(x0, y0, r0, x1, y1, r1, config): 
  
  x = x1 - x0
  y = y1 - y0
  x2y2 = x * x + y * y
  c = (x2y2 + r0 * r0 - r1 * r1) / 2.0
  d = math.sqrt( x2y2 * r0 * r0 - c * c)
  if (config == 0):
    v = np.array([(c * x + y * d) / x2y2 + x0, (c * y - x * d) / x2y2 + y0])
  else:
    v = np.array([(c * x - y * d) / x2y2 + x0, (c * y + x * d) / x2y2 + y0])
  
  return v

# Knee cylinder to angle:
def cylinderToAngleCrossKnee(cyl_len, a10, a20, b10, b20, c0):

  o = np.array([0.0, 0.0])

  a2 = circleIntersectPoint(o[0], o[1], np.linalg.norm(o - a20), c0[0], c0[1], cyl_len, 0)

  bet = math.acos ( np.dot(a2,a20) / (np.linalg.norm(o - a20) * np.linalg.norm(o - a2)))
  rot2d = np.array([[math.cos(-bet),-math.sin(-bet)],[math.sin(-bet), math.cos(-bet)]])

  a1 = rot2d @ a10

  b1 = circleIntersectPoint(b20[0], b20[1], np.linalg.norm(b10 - b20) , a1[0], a1[1], np.linalg.norm(a10 - b10), 0)

  ang = math.acos(np.dot(a10 - b10, a1 - b1) / (np.linalg.norm(a10 - b10) * np.linalg.norm(a1 - b1)))

  return ang

# Knee Jacobian
# From AngletoCylinder.cpp, lines 880-883, we can see that "j" is identical to Jj^T in our calculations
# const double len = m_qCurrent.data[RC_K_ID_HRPSYS];
# const double tau = m_tau.data[R_KNEE_P_ID_HRPSYS];
# const double j   = cylinderToAngleCrossKneeDot(len, KNEE_P_A1, KNEE_P_A2, KNEE_P_B1, KNEE_P_B2, KNEE_P_C);
# m_tau.data[RC_K_ID_HRPSYS] = j * tau; // RC-K-linear

def cylinderToAngleCrossKneeDot (cyl_len, a10, a20, b10, b20, c0):

  dL = 0.000001; #[m]
  return (cylinderToAngleCrossKnee(cyl_len+dL, a10, a20, b10, b20, c0) - cylinderToAngleCrossKnee(cyl_len, a10, a20, b10, b20, c0)) / dL

# Knee angle to cylinder:
def angleToCylinderCrossKnee(ang, a10, a20, b10, b20, c0, init_len, min_len, max_len):

  delta = 0.000001 #[m]
  thresh = 0.0001; #0.0001 [rad] = 0.0057 [deg]

  loop_num = 10
  tmp_ang = cylinderToAngleCrossKnee(init_len, a10, a20, b10, b20, c0)
  tmp_len = init_len

  for i in range(loop_num):
    tmp_len = tmp_len - (cylinderToAngleCrossKnee(tmp_len, a10, a20, b10, b20, c0) - ang) / cylinderToAngleCrossKneeDot(tmp_len, a10, a20, b10, b20, c0)

    # Case when tmp_len is NaN:
    if math.isnan(tmp_len):  
        tmp_len = init_len + (i * delta)  

    tmp_ang = cylinderToAngleCrossKnee(tmp_len, a10, a20, b10, b20, c0)

    # Break if the threshold condition is met
    if thresh > abs(ang - tmp_ang):  
        break  

    # If max iterations are reached, print an error message
    if i == loop_num - 1:
        print(f"[atc] loop_num max : tmp_len={tmp_len} tmp_ang={tmp_ang * 180 / math.pi} [deg]")

    # Clamp tmp_len within the limits
    if tmp_len > max_len:
       tmp_len = max_len - delta

    if tmp_len < min_len:
       tmp_len = min_len + delta

  return tmp_len

# Implementing Kv_j = Jj^-T*Km*kv_m*Km*Jj^-1
def Motor_to_Joint_Dgains(Kv_m, InvJj, Km):

  Kv_j = InvJj.transpose() @ Km @ Kv_m @ Km @ InvJj

  return Kv_j

# Implementing Kp_j*qj = Jj^-T * Km * Kp_m * qm. This implementation is in vector form, not in Matrix form
def Motor_to_Joint_Pgains(Kp_m, InvJj, Km, qr, qp, leftLength, rightLength):

  motors_m = np.array([Kp_m[0] * leftLength, Kp_m[1] * rightLength])

  Kp_j_joints = InvJj.transpose() @ Km  @ motors_m

  # if (qr == 0.0):

  #   Kp_j1 = 0.0

  # else:

  #   Kp_j1 = Kp_j_joints[0]/qr
  
  # if (qp == 0.0):

  #   Kp_j2 = 0.0

  # else:

  #   Kp_j2 = Kp_j_joints[1]/qp

  if (qr == 0.0):
     qr = -0.1
  if (qp == 0.0):
     qp = -0.1

  Kp_j1 = Kp_j_joints[0]/qr   
  Kp_j2 = Kp_j_joints[1]/qp

  Kp_j = np.array([Kp_j1, Kp_j2])

  return Kp_j

# Knee case: the knee has 1 joint only, so the inputs and gains aren't matrices
# Implementing Kv_j = Jj^-T*Km*kv_m*Km*Jj^-1

def Motor_to_Joint_Dgains_Knee(Kv_m, J, Km):

  Kv_j = 1/J * Km * Kv_m * Km * 1/J

  return Kv_j

# Implementing Kp_j*qj = Jj^-T * Km * Kp_m * qm
# Use kr001_cylinder_to_angle_gimbal, the cylinder lengths are the qm, while the qr and qp are joint positions.

def Motor_to_Joint_Pgains_Knee(Kp_m,  j,  Km, angle, cylinder_position):

  Kp_j = 1/j * Km * Kp_m * cylinder_position * 1/angle

  return Kp_j

##################################### Main #####################################

# Knee P

# Suppose you have arrays Kp and Kd containing the PD gains of the motor level

Kp_m_knee = 3.5
Kv_m_knee = 0.0003

# Km for knee
Km_knee = 1000.0 * 2 * math.pi * 2 * 0.36

# cross knee settings
KNEE_P_A1 = np.array([0.0290, -0.0450]) #[m]
KNEE_P_A2 = np.array([0.0500,  0.0710]) #[m]
KNEE_P_B1 = np.array([0.0000, -0.0590]) #[m]
KNEE_P_B2 = np.array([0.0270,  0.0000]) #[m]
KNEE_P_C = np.array([0.0500,  0.2650]) #[m]
KNEE_P_INIT_LEN = 0.1941 #[m]
KNEE_P_MIN_LEN  = 0.1940 #[m]
KNEE_P_MAX_LEN  = 0.346398 #[m]

# Joint range for the knee: 0<p<150, half-sitting is 36.0

# Arrays for plotting:

joints_data = []
Kp_knee_data = []
Kv_knee_data = []

for i in np.arange(1.0, 150.0, 5.0):

    THP_knee = math.radians(i)

    #Cylinder position
    length = angleToCylinderCrossKnee(THP_knee,KNEE_P_A1,KNEE_P_A2,KNEE_P_B1,KNEE_P_B2,KNEE_P_C,KNEE_P_INIT_LEN,KNEE_P_MIN_LEN,KNEE_P_MAX_LEN)

    #Jacobian
    j_knee = cylinderToAngleCrossKneeDot(length,KNEE_P_A1,KNEE_P_A2,KNEE_P_B1,KNEE_P_B2,KNEE_P_C)

    # Calculating kp_j and Kv_j
    Kp_j_knee = Motor_to_Joint_Pgains_Knee(Kp_m_knee, j_knee, Km_knee, THP_knee, length)

    Kv_j_knee = Motor_to_Joint_Dgains_Knee(Kv_m_knee, j_knee, Km_knee)

    # print("Km:\n ", Km_knee)
    # print("Jacobian:\n ", j_knee)
    # print("Kv at the Knee P level:\n ", Kv_j_knee)
    # print("Kp at the Knee P level:\n ", Kp_j_knee)

    joints_data.append(i)
    Kp_knee_data.append(Kp_j_knee)
    Kv_knee_data.append(Kv_j_knee)

# Plots

plt.figure()
plt.plot(joints_data, Kp_knee_data, label = "P gains knee_P", c = 'blue')
plt.scatter(joints_data, Kp_knee_data, label = "P gains knee_P", c = 'red')
plt.xlabel("Knee Joint Position")
plt.ylabel("kp_j_knee_P")
plt.title("knee P joint level P gains as function of knee P joint positions")
plt.legend()
plt.grid()

plt.figure()
plt.plot(joints_data, Kv_knee_data, label = "D gains knee_P", c = 'blue')
plt.scatter(joints_data, Kv_knee_data, label = "D gains knee_P", c = 'red')
plt.xlabel("Knee Joint Position")
plt.ylabel("kv_j_knee_P")
plt.title("knee P joint level D gains as function of knee P joint positions")
plt.legend()
plt.grid()
# plt.show()


#--------------------------------------------------------------------------------------------------------------------------------

# Crotch P/R 

# Defining element-level motor P gains:
Kp_m_crotch = [0.95, 1.06]

# Defining Kv_m as a matrix
Kv_m_crotch = [[0.0002, 0], [0, 0.0002]]

# Km for crotch
Km_crotch = np.array([[1000 * 2 * math.pi * 0.454545455, 0], [0 ,1000 * 2 * math.pi * 0.454545455]])

# 15 double parameters: THR, THP, LT, LC, L1Y, L1Z, L2X, L2Y, L2Z, L3X, L3Z, LTxLT, LCxLC, LCx2, QOFF2

# 股 P/R 

gm_Crotch = GimbalMethod.ROLL_PITCH

CROTCH_ROD_MIN_LENGTH     = 0.24157 
CROTCH_ROD_MAX_LENGTH     = 0.35480 
CROTCH_ROD_DEFAULT_LENGTH = 0.26838
CROTCH_R_DEFAULT_ANGLE   = math.radians(0.0)
CROTCH_P_DEFAULT_ANGLE   = math.radians(0.0) 

CROTCH_LT  =  0.08000 
CROTCH_LC  =  0.07000 
CROTCH_L1Y =  0.04500
CROTCH_L1Z =  0.05000
CROTCH_L2X =  0.04924
CROTCH_L2Y =  0.04500
CROTCH_L2Z = -0.00868
CROTCH_L3X =  0.00000
CROTCH_L3Z =  0.30000
CROTCH_QOFF2 =  math.radians(0.0)

CROTCH_LTxLT = CROTCH_LT * CROTCH_LT
CROTCH_LCxLC = CROTCH_LC * CROTCH_LC
CROTCH_LCx2  = CROTCH_LC * 2.0

# For the crotch: -110<p<30, -30<r<30
# According to the AngleToCylinder.cpp, the kr001_angle_to_cylinder_gimbal (hence the jacobian too) take -qp (pitch) and qr (roll)
# To calculate Kp, qp is used for the joint positions

crotch_roll_data = []
crotch_pitch_data = []
Kp_crotch_roll_data = []
Kp_crotch_pitch_data = []
Kv_crotch_roll_data = []
Kv_crotch_pitch_data = []

THR_crotch = math.radians(-30)

for i in np.arange(-110.0, 30.01, 10.0):

  qp = math.radians(i)
  THP_crotch = -qp

  leftlength_crotch, rightlength_crotch = kr001_angle_to_cylinder_gimbal(gm_Crotch, THR_crotch, THP_crotch,
                                CROTCH_LT, CROTCH_LC, CROTCH_L1Y, CROTCH_L1Z,
                                CROTCH_L2X, CROTCH_L2Y, CROTCH_L2Z, CROTCH_L3X, CROTCH_L3Z,
                                CROTCH_LTxLT, CROTCH_LCxLC, CROTCH_LCx2, CROTCH_QOFF2)


  InvJj_crotch = Jacobian_MotorTorque_CylinderForce(gm_Crotch,THR_crotch,THP_crotch,CROTCH_LT,CROTCH_LC,CROTCH_L1Y,CROTCH_L1Z,
                                  CROTCH_L2X,CROTCH_L2Y,CROTCH_L2Z,CROTCH_L3X,CROTCH_L3Z,
                                  CROTCH_LTxLT,CROTCH_LCxLC,CROTCH_LCx2,CROTCH_QOFF2)
  

  # Calculating kp_j and Kv_j

  Kp_j_crotch = Motor_to_Joint_Pgains(Kp_m_crotch, InvJj_crotch, Km_crotch, THR_crotch, qp, leftlength_crotch, rightlength_crotch)
  Kv_j_crotch = Motor_to_Joint_Dgains(Kv_m_crotch, InvJj_crotch, Km_crotch)

  # cnt = 0
  # for rows in Kv_j_crotch:
  #  print(rows)
  #  cnt+=1
  #  if cnt%2 == 0:
  #     print("")

  crotch_pitch_data.append(i)
  crotch_roll_data.append(math.degrees(THR_crotch))
  Kp_crotch_roll_data.append(Kp_j_crotch[0])
  Kp_crotch_pitch_data.append(Kp_j_crotch[1])
  Kv_crotch_roll_data.append(Kv_j_crotch[0][0])
  Kv_crotch_pitch_data.append(Kv_j_crotch[1][1])

  THR_crotch = THR_crotch + math.radians(4)

# print("Kp at the Crotch R level:\n ", Kp_crotch_roll_data)
# print("")
# print("Kp at the Crotch P level:\n ", Kp_crotch_pitch_data)
# print("\n")
# print("Crotch R:\n ", crotch_roll_data)
# print("")
# print("Crotch P:\n ", crotch_pitch_data)
# print("Kv at the Crotch R level:\n ", Kv_crotch_roll_data)
# print("Kv at the Crotch P level:\n ", Kv_crotch_pitch_data)

# Plots

# Simple 2D plot
# plt.figure()
# plt.plot(crotch_pitch_data, Kp_crotch_pitch_data, label = "P gains crotch_P", c = 'blue')
# plt.scatter(crotch_pitch_data, Kp_crotch_pitch_data, label = "P gains crotch_P", c = 'red')
# plt.xlabel("Crotch_P Joint Position")
# plt.ylabel("kp_j_crotch_P")
# plt.title("Crotch_P joint level P gains as function of Crotch_P joint positions")
# plt.legend()
# plt.grid()
# plt.show()

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection = '3d')
ax.scatter(np.array(crotch_pitch_data), np.array(crotch_roll_data), np.array(Kp_crotch_roll_data), c = 'orange', label = 'P gain for Crotch_R')
ax.plot(np.array(crotch_pitch_data), np.array(crotch_roll_data), np.array(Kp_crotch_roll_data), c = 'brown', label = 'P gain for Crotch_R')
ax.scatter(np.array(crotch_pitch_data), np.array(crotch_roll_data), np.array(Kp_crotch_pitch_data), c = 'red', label = 'P gain for Crotch_P')
ax.plot(np.array(crotch_pitch_data), np.array(crotch_roll_data), np.array(Kp_crotch_pitch_data), c = 'blue', label = 'P gain for Crotch_P')
ax.set_xlabel("Crotch_P Joint Position")
ax.set_ylabel("Crotch_R Joint Position")
ax.set_zlabel("kp_j_crotch")
ax.set_title("Crotch_P joint level P gains as function of Crotch_P/R joint positions")
ax.legend()
ax.grid()

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection = '3d')
ax.scatter(np.array(crotch_pitch_data), np.array(crotch_roll_data), np.array(Kv_crotch_roll_data), c = 'orange', label = 'D gain for Crotch_R')
ax.plot(np.array(crotch_pitch_data), np.array(crotch_roll_data), np.array(Kv_crotch_roll_data), c = 'brown', label = 'D gain for Crotch_R')
ax.scatter(np.array(crotch_pitch_data), np.array(crotch_roll_data), np.array(Kv_crotch_pitch_data), c = 'red', label = 'D gain for Crotch_P')
ax.plot(np.array(crotch_pitch_data), np.array(crotch_roll_data), np.array(Kv_crotch_pitch_data), c = 'blue', label = 'D gain for Crotch_P')
ax.set_xlabel("Crotch_P Joint Position")
ax.set_ylabel("Crotch_R Joint Position")
ax.set_zlabel("Kv_j_crotch")
ax.set_title("Crotch_P joint level P gains as function of Crotch_P/R joint positions")
ax.legend()
ax.grid()
# plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

# Ankle P/R 

# Defining element-level motor P gains:
Kp_m_ankle = [0.3, 0.3]

# Defining Kv_m as a matrix
Kv_m_ankle = [[0.0001, 0], [0, 0.0001]]

# Km for ankle
Km_ankle = np.array([[1000 * 2 * math.pi * 0.454545455, 0], [0 ,1000 * 2 * math.pi * 0.454545455]])

# 15 double parameters: THR, THP, LT, LC, L1Y, L1Z, L2X, L2Y, L2Z, L3X, L3Z, LTxLT, LCxLC, LCx2, QOFF2

# 股 P/R 

gm_ankle = GimbalMethod.PITCH_ROLL

ANKLE_ROD_MIN_LENGTH     = 0.24004 
ANKLE_ROD_MAX_LENGTH     = 0.35480 
ANKLE_ROD_DEFAULT_LENGTH = 0.288701
ANKLE_R_DEFAULT_ANGLE   = math.radians(0.0)
ANKLE_P_DEFAULT_ANGLE   = math.radians(0.0) 

ANKLE_LT  =  0.08000 
ANKLE_LC  =  0.07000 
ANKLE_L1Y =  0.03700
ANKLE_L1Z =  0.05000
ANKLE_L2X =  0.04924
ANKLE_L2Y =  0.03700
ANKLE_L2Z = 0.00868
ANKLE_L3X =  0.00000
ANKLE_L3Z =  0.30000
ANKLE_QOFF2 =  math.radians(0.0)

ANKLE_LTxLT = ANKLE_LT * ANKLE_LT
ANKLE_LCxLC = ANKLE_LC * ANKLE_LC
ANKLE_LCx2  = ANKLE_LC * 2.0

# For the ankle: -94<p<50, -32<r<32
# According to the AngleToCylinder.cpp, the kr001_angle_to_cylinder_gimbal (hence the jacobian too) take -qp (pitch) and qr (roll)
# To calculate Kp, qp is used for the joint positions

ankle_roll_data = []
ankle_pitch_data = []
Kp_ankle_roll_data = []
Kp_ankle_pitch_data = []
Kv_ankle_roll_data = []
Kv_ankle_pitch_data = []

THR_ankle = math.radians(-32)

for i in np.arange(-90.0, 50.01, 10.0):

  qp = math.radians(i)
  THP_ankle = -qp

  leftlength_ankle, rightlength_ankle = kr001_angle_to_cylinder_gimbal(gm_ankle, THR_ankle, THP_ankle,
                                ANKLE_LT, ANKLE_LC, ANKLE_L1Y, ANKLE_L1Z,
                                ANKLE_L2X, ANKLE_L2Y, ANKLE_L2Z, ANKLE_L3X, ANKLE_L3Z,
                                ANKLE_LTxLT, ANKLE_LCxLC, ANKLE_LCx2, ANKLE_QOFF2)


  InvJj_ankle = Jacobian_MotorTorque_CylinderForce(gm_ankle,THR_ankle,THP_ankle,ANKLE_LT,ANKLE_LC,ANKLE_L1Y,ANKLE_L1Z,
                                  ANKLE_L2X,ANKLE_L2Y,ANKLE_L2Z,ANKLE_L3X,ANKLE_L3Z,
                                  ANKLE_LTxLT,ANKLE_LCxLC,ANKLE_LCx2,ANKLE_QOFF2)
  

  # Calculating kp_j and Kv_j

  Kp_j_ankle = Motor_to_Joint_Pgains(Kp_m_ankle, InvJj_ankle, Km_ankle, THR_ankle, qp, leftlength_ankle, rightlength_ankle)
  Kv_j_ankle = Motor_to_Joint_Dgains(Kv_m_ankle, InvJj_ankle, Km_ankle)

  # cnt = 0
  # for rows in Kv_j_ankle:
  #  print(rows)
  #  cnt+=1
  #  if cnt%2 == 0:
  #     print("")

  ankle_pitch_data.append(i)
  ankle_roll_data.append(math.degrees(THR_ankle))
  Kp_ankle_roll_data.append(Kp_j_ankle[0])
  Kp_ankle_pitch_data.append(Kp_j_ankle[1])
  Kv_ankle_roll_data.append(Kv_j_ankle[0][0])
  Kv_ankle_pitch_data.append(Kv_j_ankle[1][1])

  THR_ankle = THR_ankle + math.radians(4.5)

# print("Kp at the ankle R level:\n ", Kp_ankle_roll_data)
# print("")
# print("Kp at the ankle P level:\n ", Kp_ankle_pitch_data)
# print("\n")
# print("ankle R:\n ", ankle_roll_data)
# print("")
# print("ankle P:\n ", ankle_pitch_data)
# print("Kv at the ankle R level:\n ", Kv_ankle_roll_data)
# print("Kv at the ankle P level:\n ", Kv_ankle_pitch_data)

# Plots

# Simple 2D plot
# plt.figure()
# plt.plot(ankle_pitch_data, Kp_ankle_pitch_data, label = "P gains ankle_P", c = 'blue')
# plt.scatter(ankle_pitch_data, Kp_ankle_pitch_data, label = "P gains ankle_P", c = 'red')
# plt.xlabel("ankle_P Joint Position")
# plt.ylabel("kp_j_ankle_P")
# plt.title("ankle_P joint level P gains as function of ankle_P joint positions")
# plt.legend()
# plt.grid()
# plt.show()

fig3 = plt.figure()
ax = fig3.add_subplot(111, projection = '3d')
ax.scatter(np.array(ankle_pitch_data), np.array(ankle_roll_data), np.array(Kp_ankle_roll_data), c = 'orange', label = 'P gain for ankle_R')
ax.plot(np.array(ankle_pitch_data), np.array(ankle_roll_data), np.array(Kp_ankle_roll_data), c = 'brown', label = 'P gain for ankle_R')
ax.scatter(np.array(ankle_pitch_data), np.array(ankle_roll_data), np.array(Kp_ankle_pitch_data), c = 'red', label = 'P gain for ankle_P')
ax.plot(np.array(ankle_pitch_data), np.array(ankle_roll_data), np.array(Kp_ankle_pitch_data), c = 'blue', label = 'P gain for ankle_P')
ax.set_xlabel("ankle_P Joint Position")
ax.set_ylabel("ankle_R Joint Position")
ax.set_zlabel("kp_j_ankle")
ax.set_title("ankle_P joint level P gains as function of ankle_P/R joint positions")
ax.legend()
ax.grid()

fig4 = plt.figure()
ax = fig4.add_subplot(111, projection = '3d')
ax.scatter(np.array(ankle_pitch_data), np.array(ankle_roll_data), np.array(Kv_ankle_roll_data), c = 'orange', label = 'D gain for ankle_R')
ax.plot(np.array(ankle_pitch_data), np.array(ankle_roll_data), np.array(Kv_ankle_roll_data), c = 'brown', label = 'D gain for ankle_R')
ax.scatter(np.array(ankle_pitch_data), np.array(ankle_roll_data), np.array(Kv_ankle_pitch_data), c = 'red', label = 'D gain for ankle_P')
ax.plot(np.array(ankle_pitch_data), np.array(ankle_roll_data), np.array(Kv_ankle_pitch_data), c = 'blue', label = 'D gain for ankle_P')
ax.set_xlabel("ankle_P Joint Position")
ax.set_ylabel("ankle_R Joint Position")
ax.set_zlabel("Kv_j_ankle")
ax.set_title("ankle_P joint level P gains as function of ankle_P/R joint positions")
ax.legend()
ax.grid()
plt.show()