import numpy as np
from enum import Enum
import math
import matplotlib.pyplot as plt

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

    if GimbalMethod == 1: #ROLL_PITCH
        # |  cos(qp)  0   sin(qp) |   |  1,        0,        0 |   | -L2X |
        # |        0  1        0  | * |  0,  cos(qr), -sin(qr) | * | -L2Y |
        # | -sin(qp)  0   cos(qp) |   |  0,  sin(qr),  cos(qr) |   |  L2Z |
       cqrL2Z_sqrL2Y = (cqrL2Z - sqrL2Y) 
       phx = -cqpL2X + sqp * cqrL2Z_sqrL2Y
       phy = -cqrL2Y - sqrL2Z
       phz =  sqpL2X + cqp * cqrL2Z_sqrL2Y
    elif GimbalMethod == 0: #PITCH_ROLL
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
    if GimbalMethod == 1: #ROLL_PITCH
        # |  cos(qp)  0   sin(qp) |   |  1,        0,        0 |   | -L2X |
        # |        0  1        0  | * |  0,  cos(qr), -sin(qr) | * | +L2Y |
        # | -sin(qp)  0   cos(qp) |   |  0,  sin(qr),  cos(qr) |   |  L2Z |
       cqrL2Z_sqrL2Y = (cqrL2Z + sqrL2Y) 
       phx = -cqpL2X + sqp * cqrL2Z_sqrL2Y
       phy = cqrL2Y - sqrL2Z
       phz =  sqpL2X + cqp * cqrL2Z_sqrL2Y
    elif GimbalMethod == 0: #PITCH_ROLL
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

    # Case when tmp_len is NaN,:
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
def Motor_to_Joint_Pgains(InvJj, Km, qr, qp, leftLength, rightLength):

 # Defining element-level motor P gains:
  Kp_m1 = 0.9
  Kp_m2 = 0.9

  joints_m = np.array([qr, qp])

  motors_m = np.array([Kp_m1 * leftLength, Kp_m2 * rightLength])

  Kp_j_joints = InvJj.transpose() @ Km  @ motors_m

  if (qr == 0.0):

    Kp_j1 = 0.0

  else:

    Kp_j1 = Kp_j_joints[0]/qr
  
  if (qp == 0.0):

    Kp_j2 = 0.0

  else:

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

def Motor_to_Joint_Pgains_Knee( Kp_m,  j,  Km, angle, cylinder_position):

  Kp_j = 1/j * Km * Kp_m * cylinder_position * 1/angle

  return Kp_j

##################################### Main #####################################

# Knee P

#Suppose you have arrays Kp and Kd containing the PD gains of the motor level

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
plt.show()




