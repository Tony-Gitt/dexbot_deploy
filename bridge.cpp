// add
#include "humanoid.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <unistd.h> // 用于 isatty 和 getchar
#include <termios.h>
#include <sys/select.h>
#include <chrono>
#include <cmath>
#include <signal.h>   // SIGINT、signal()

#include <sched.h> 
#include <sys/timerfd.h>  // 包含 timerfd_create、timerfd_settime 等函数声明
#include <chrono>
#include <cstdint>        // 包含 uint64_t 类型定义
// #include "joystick_xbox.h"  // 2025.12.1
using namespace std;
// 全局变量
bool is_initial_transition_done = false;    // 标记初始平滑过渡是否完成
bool start_model_inference = false; // 是否开始模型推理的标志
bool start_control_send = false; // 是否开始控制发送的标志
float command_x = 0; // 键盘命令
float command_y = 0; // 键盘命令
float command_yaw = 0; // 键盘命令
bool command_stand = false; // 
bool still_flag = false;
bool promptPrinted = false;

//////////////////////////////////////////////////////////////////

int32_t Kp_walk[7] = {70,70,78,105,31,31,0};
int32_t Kd_walk[7] = {307,307,345,460,138,138,0};
int32_t Kp_waist[7] = {0,0,140,0,0,0, 0};
int32_t Kd_waist[7] = {0,0,306,0,0,0,0};
int32_t Kp_stand[7] = {1000,1000,1000,1000,1000,1000, 0};
int32_t Kd_stand[7] = {1000,1000,1000,1000,1000,1000, 0}; 
// int32_t Kp_arm_l[7] = {50,800,800,800,800,0, 0};
// int32_t Kd_arm_l[7] = {80,700,700,700,700,0,0}; 

int32_t Kp_arm[7] = {15,15,15,15,15,15,15};
int32_t Kd_arm[7] = {24,24,24,24,24,24,24};  
int32_t usd2urdf[27] = {6, 0, 12, 7, 1, 20, 13, 8, 2, 21, 14, 9, 3, 22, 15, 10, 4, 23, 16, 11, 5, 24, 17, 25, 18, 26, 19};

float default_angle[25] = {-0.35, 0.06, 0.18, 0.72, -0.4, 0.0,
                           -0.35, -0.06, -0.18, 0.72, -0.4, 0.0,
                            0.0,
                            -0, -0.1, 0, 0, 0.0, 0.0, 
                            -0, 0.1, 0, 0, 0.0, 0.0};

// float default_angle[27] = {-0.35, 0.06, 0.18, 0.72, -0.4, 0.0,
//                            -0.35, -0.06, -0.18, 0.72, -0.4, 0.0,
//                             0.0,
//                             -0.04, -0.40, 0.13, -1.38, 0.0, 0.0, 0.0 
//                             -0.04, 0.40, 0.13, 1.38, 0.0, 0.0, 0.0};

float Iq_torque[12] = {75, 75, 100, 75, 100, 100,
                        75, 75, 100, 75, 100, 100};
// 初始化关节角度####
std::vector<float> init_joint_act = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0}; // 初始关节角度
std::vector<float> joint_act = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0};
std::vector<float> joint_act_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0};
std::vector<float> joint_action = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0};
std::vector<float> target_dof_pos = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

// 新增全局变量，用于存储 tau 值的历史记录
std::vector<std::vector<float>> tau_history;
std::vector<std::vector<float>> inputs_history;
std::vector<std::vector<float>> outputs_history;
std::vector<std::vector<float>> dof_torque_history; 
std::vector<std::vector<float>> dof_pos_history;
std::vector<std::vector<float>> dof_vel_history;
std::vector<std::vector<float>> desired_Iq_history;
std::mutex tau_history_mutex;
std::mutex dof_torque_history_mutex;
std::mutex dof_pos_send_mutex;
std::mutex dof_pos_history_mutex;
std::mutex dof_vel_history_mutex;
std::mutex desired_Iq_history_mutex;


float LeftcurAngle[7], RightcurAngle[7], curVel[7], curAcc[7], LeftfeedbackTorque[7], RightfeedbackTorque[7];
float LeftcurAngle_arm[7], RightcurAngle_arm[7], curVel_arm[7], curAcc_arm[7];
float WaistAngle[7];
// std::vector<float> action_store(12, 0.0f);
// ActionBuffer<ActionState> action_buffer_;

std::mutex mtx; // 用于线程间同步
std::condition_variable cv_cond; // 用于线程间通信
std::vector<float> last_actions(27, 0.0f); // 定义
std::vector<float> last_actions_ang(27, 0.0f); // 定义
std::atomic<bool> running{true}; // 控制线程运行状态
std::atomic<bool> smooth{true};
// auto loop_start = std::chrono::high_resolution_clock::now();

bool first_update = false; // 标志位，用于区分第一次更新
int count_dt = 0;
int tt=0;
std::deque<torch::Tensor> history_inputs;
// ************************ 新增函数 **********//


// ****************************************************//


void HumanoidController::stopThreads() {
    running = false; // 设置标志位为 false，停止线程
    cv_cond.notify_all(); // 唤醒所有等待的线程

    if (policy_thread.joinable()) {
        policy_thread.join(); // 等待策略更新线程结束
        
    }
    if (control_thread.joinable()) {
        control_thread.join(); // 等待控制发送线程结束
    }
}

// 在程序结束时保存 tau 和 dof_torque 值到 CSV 文件
void save_data_to_csv() {
    // 保存 tau 值到 CSV 文件
    std::ofstream tau_csv_file("tau_history.csv");
    if (tau_csv_file.is_open()) {
        // 写入表头
        for (int i = 0; i  < 12; ++i) {
            tau_csv_file << "Joint " << i << " Desired Torque zjl";
            if (i != 11) tau_csv_file << ",";
        }
        tau_csv_file << "\n";

        // 写入数据
        for (const auto& tau_row : tau_history) {
            for (size_t i = 0; i < tau_row.size(); ++i) {
                tau_csv_file << tau_row[i];
                if (i != tau_row.size() - 1) tau_csv_file << ",";
            }
            tau_csv_file << "\n";
        }
        tau_csv_file.close();
        std::cout << "Tau history saved to tau_history.csv\n";
    } else {
        std::cerr << "Failed to open tau_history.csv for writing\n";
    }

    // 保存 dof_torque 值到 CSV 文件
    std::ofstream dof_torque_csv_file("dof_torque_history.csv");
    if (dof_torque_csv_file.is_open()) {
        // 写入表头
        for (int i = 0; i < 12; ++i) {
            dof_torque_csv_file << "Joint " << i << " Torque zjl";
            if (i != 11) dof_torque_csv_file << ",";
        }
        dof_torque_csv_file << "\n";

        // 写入数据
        for (const auto& torque_row : dof_torque_history) {
            for (size_t i = 0; i < torque_row.size(); ++i) {
                dof_torque_csv_file << torque_row[i];
                if (i != torque_row.size() - 1) dof_torque_csv_file << ",";
            }
            dof_torque_csv_file << "\n";
        }
        dof_torque_csv_file.close();
        std::cout << "DOF Torque history saved to dof_torque_history.csv\n";
    } else {
        std::cerr << "Failed to open dof_torque_history.csv for writing\n";
    }

    // 保存 dof_pos 值到 CSV 文件
    std::ofstream dof_position_csv_file("dof_pos_history.csv");
    if (dof_position_csv_file.is_open()) {
        // 写入表头
        for (int i = 0; i < 12; ++i) {
            dof_position_csv_file << "Joint " << i << " Position zjl";
            if (i != 11) dof_position_csv_file << ",";
        }
        dof_position_csv_file << "\n";

        // 写入数据
        for (const auto& torque_row : dof_pos_history) {
            for (size_t i = 0; i < torque_row.size(); ++i) {
                dof_position_csv_file << torque_row[i];
                if (i != torque_row.size() - 1) dof_position_csv_file << ",";
            }
            dof_position_csv_file << "\n";
        }
        dof_position_csv_file.close();
        std::cout << "DOF Torque history saved to dof_pos_history.csv\n";
    } else {
        std::cerr << "Failed to open dof_pos_history.csv for writing\n";
    }

    // 保存 dof_vel 值到 CSV 文件
    std::ofstream dof__velocity_csv_file("dof_vel_history.csv");
    if (dof__velocity_csv_file.is_open()) {
        // 写入表头
        for (int i = 0; i < 15; ++i) {
            dof__velocity_csv_file << "Joint " << i << " Velocity zjl";
            if (i != 14) dof__velocity_csv_file << ",";
        }
        dof__velocity_csv_file << "\n";

        // 写入数据
        for (const auto& torque_row : dof_vel_history) {
            for (size_t i = 0; i < torque_row.size(); ++i) {
                dof__velocity_csv_file << torque_row[i];
                if (i != torque_row.size() - 1) dof__velocity_csv_file << ",";
            }
            dof__velocity_csv_file << "\n";
        }
        dof__velocity_csv_file.close();
        std::cout << "DOF Torque history saved to dof_vel_history.csv\n";
    } else {
        std::cerr << "Failed to open dof_vel_history.csv for writing\n";
    }

    // 保存 desired_Iq 值到 CSV 文件
    std::ofstream desired_Iq_csv_file("desired_Iq_history.csv");
    if (desired_Iq_csv_file.is_open()) {
        // 写入表头
        for (int i = 0; i < 12; ++i) {
            desired_Iq_csv_file << "Joint " << i << " desired_Iq zjl";
            if (i != 11) desired_Iq_csv_file << ",";
        }
        desired_Iq_csv_file << "\n";

        // 写入数据
        for (const auto& torque_row : desired_Iq_history) {
            for (size_t i = 0; i < torque_row.size(); ++i) {
                desired_Iq_csv_file << torque_row[i];
                if (i != torque_row.size() - 1) desired_Iq_csv_file << ",";
            }
            desired_Iq_csv_file << "\n";
        }
        desired_Iq_csv_file.close();
        std::cout << "DOF Torque history saved to desired_Iq_history.csv\n";
    } else {
        std::cerr << "Failed to open desired_Iq_history.csv for writing\n";
    }
}

// 在 humanoid.cpp 中定义析构函数
HumanoidController::~HumanoidController() {
    stopThreads(); // 停止线程
    if (policy_thread.joinable()) {
        policy_thread.join();
    }
    if (control_thread.joinable()) {
        control_thread.join();
    }
    // 保存 tau 值到 CSV
    save_data_to_csv();
}


////////////////////////////////////////////////////////////////////////////

// 构造函数
HumanoidController::HumanoidController(Bridge* bridge_ptr) 
    : bridge_(bridge_ptr)
{
    base_ang_vel.setZero();
    base_euler_xyz.setZero();

    // 初始化关节角度限制####
    act_pos_low = { -2.878, -0.523, -2.791, -2.355, -1.046, -0.261, -2.878, -3.663, -2.791, -2.355, -1.046, -0.261 , -2, -2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2};
    act_pos_high = { 2.878,  3.663,  2.791,  2.355,  0.436,  0.261,  2.878,  0.523,  2.791,  2.355,  0.436, 0.261 , 2,2,2,2,2,2,2,2,2,2,2,2,2};
    for (int i = 0; i < 25; i++) {
                act_pos_low[i] = (act_pos_low[i]) * 180 / M_PI; 
                act_pos_high[i] = (act_pos_high[i]) * 180 / M_PI; 
            }
    // 确保 joint_act 的大小与 act_pos_low 和 act_pos_high 一致####
    assert(joint_act.size() == act_pos_low.size());
    assert(joint_act.size() == act_pos_high.size());

}

// 初始化腿部和腕部
bool HumanoidController::initLegsAndWrist()
{
    return bridge_->InitLegsAndWrist();
}

// 获取腿部和腕部输出
bool HumanoidController::getLegsAndWristOutput(Interface::LegsAndWristOutput &output)
{
    return bridge_->getLegsAndWristOutput(output);
}

// 设置腿部和腕部输入
bool HumanoidController::setLegsAndWristInput(Interface::LegsAndWristInput &input)
{
    return bridge_->setLegsAndWristInput(input);
}

// 初始化所有硬件接口
void HumanoidController::InitAll()
{
    bridge_->InitImu();
    
    bridge_->InitQDDLegsAndWaist();

    bridge_->InitRightArm();
    bridge_->InitLeftArm();
    printf("init all111\n");
}


bool HumanoidController::AddCheckToCmd(Interface::ArmCommand &cmd)
{
    if (cmd.paraLen > 251 * 4)
        return false;
    uint32_t result = 0;
    result += cmd.cmdNum;
    result += cmd.cmdKind;
    result += cmd.cmdCode;
    result += cmd.paraLen;
    uint32_t len = cmd.paraLen / 4;
    if (len * 4 != cmd.paraLen)
        len = len + 1;

    for (int32_t i = 0; i < len; i++)
    {
        result += cmd.data[i];
    }
    cmd.data[len] = result;
    return true;
}

template<typename T>
T clip(const T& value, const  T& min_value, const T& max_value){
    return std::max(min_value,std::min(value,max_value));
}

void HumanoidController::RTmultidoftrajplanModeWithDynaKpKdTorquecmd(float desiredJoint[PLANDOF],float desiredVel[PLANDOF],float desiredAcc[PLANDOF],int32_t kp[PLANDOF],
                                                 int32_t kd[PLANDOF],float feedbackTorque[PLANDOF],int flag)
    {
    RTMultiDofTrajPlan_DynamicKpKd_TorqueFeed_Cmd cmd;
    Interface::ArmCommand oldcmd;
    Input.cmdCode = 0x00000C01;
    Input.cmdKind = 0x1111;
    Input.paraLen = 84+28+56;
    for(int i=0;i<7;i++)
    {
        cmd.desiredJoint[i]=desiredJoint[i];
        cmd.desiredVel[i]=desiredVel[i];
        cmd.desiredAcc[i]=desiredAcc[i];
        cmd.feedTorque[i] = feedbackTorque[i];
        cmd.kp[i]=kp[i];
        cmd.kd[i]=kd[i];
    }
        if (flag == RIGHTARM)
    {
        bridge_->getRightArmCommand(oldcmd);
        Input.cmdNum = oldcmd.cmdNum + 1;
    }
    else if (flag == LEFTARM)
    {
        bridge_->getLeftArmCommand(oldcmd);
        Input.cmdNum = oldcmd.cmdNum + 1;
    }
    else if(flag ==   QDDLEFTLEG){
        bridge_->getQDDLegsAndWaistCommand1(oldcmd);
        Input.cmdNum=oldcmd.cmdNum+1;
        Input.cmdCode |= 0xA0000000;
    }
    else if(flag ==   QDDRIGHTLEG){
        bridge_->getQDDLegsAndWaistCommand3(oldcmd);
        Input.cmdNum=oldcmd.cmdNum+1;
        Input.cmdCode |= 0xB0000000;
    }
    else if(flag ==   QDDWAIST){
        bridge_->getQDDLegsAndWaistCommand2(oldcmd);
        Input.cmdNum=oldcmd.cmdNum+1;
        Input.cmdCode |= 0xC0000000;
    }
    else{
        ;
    }
    memcpy(&Input.data[0],&cmd,sizeof(RTMultiDofTrajPlan_DynamicKpKd_TorqueFeed_Cmd));
    if(!AddCheckToCmd(Input)){
        return;
    }else{
        if(flag == QDDLEFTLEG){
            bridge_->setQDDLegsAndWaistCommand1(Input);
        }
        else if(flag == QDDRIGHTLEG){
            bridge_->setQDDLegsAndWaistCommand3(Input);
        }
        else if(flag == QDDWAIST){
            bridge_->setQDDLegsAndWaistCommand2(Input);
        }
        else if (flag == RIGHTARM)
        {
            bridge_->setRightArmCommand(Input);
        }
        else if (flag == LEFTARM)
        {
            bridge_->setLeftArmCommand(Input);
        }
        else{
            ;
        }
    }
    return;
}


std::vector<float> HumanoidController::Init_Position()
{
    std::vector<float> dof_pos(25);
    ros::Time now = ros::Time::now();
    for (int i = 0; i < 12; ++i)
    {
        bridge_->getQDDLegsAndWaistInfo(LWInfo);
        // 腿上传回数据为6+3+6，其中腰部最后一个为真实数据
        if (i < 6)
        {
            dof_pos[i] = LWInfo.realJointPos[i + 9];
            // dof_pos[i] = 0;
        }
        else
        {
            dof_pos[i] = LWInfo.realJointPos[i - 6];
            // dof_pos[i] = 0;
        }
        // std::cout << "Position: (" << dof_pos[i] << ", " << ")\n";
        std::cout << "Position[" << i << "]: " << dof_pos[i] 
              << " (Time: " << now.sec << "." << std::setfill('0') 
              << std::setw(9) << now.nsec << ")\n";
    }
    for (int i = 0; i < 12; ++i)
    {
        bridge_->getLeftArmInformation(LeftArmOutput);
        bridge_->getRightArmInformation(RightArmOutput);
       if (i < 6)
        {
            dof_pos[i+13] = RightArmOutput.realJointPos[i];
          
        }else{

            dof_pos[i+13] = LeftArmOutput.realJointPos[i-6];
        }
        std::cout << "Position_arm[" << i << "]: " << dof_pos[i+12] 
        << " (Time: " << now.sec << "." << std::setfill('0') 
        << std::setw(9) << now.nsec << ")\n";
    }
    
    bridge_->getQDDLegsAndWaistInfo(LWInfo);
    dof_pos[12] = LWInfo.realJointPos[8];
    std::cout << "Position["<< 24 << "]: " << dof_pos[24] 
              << " (Time: " << now.sec << "." << std::setfill('0') 
              << std::setw(9) << now.nsec << ")\n";

    return dof_pos;
}
// 2. position 12d
std::vector<float> HumanoidController::Position()
{
    std::vector<float> dof_pos(27);
    for (int i = 0; i < 13; ++i)
    {
        bridge_->getQDDLegsAndWaistInfo(LWInfo);
        if (i < 6)
        {
            dof_pos[i] = LWInfo.realJointPos[i + 9] * M_PI / 180 - default_angle[i];
            // dof_pos[i] = 0;
        }
        else if (i < 12)
        {
            dof_pos[i] = LWInfo.realJointPos[i - 6] * M_PI / 180 - default_angle[i];
            // dof_pos[i] = 0;
        }
        else
        {
            dof_pos[i] = LWInfo.realJointPos[8] * M_PI / 180 - default_angle[i];
            // dof_pos[i] = 0;
        }
        // std::cout << "Position: (" << dof_pos[i] << ", " << ")\n";
    }
    for (int i = 0; i < 12; ++i)
    {
        bridge_->getLeftArmInformation(LeftArmOutput);
        bridge_->getRightArmInformation(RightArmOutput);
        if (i < 6)
        {
            dof_pos[i+13] = RightArmOutput.realJointPos[i] * M_PI / 180 - default_angle[i+13];
          
        }else{

            dof_pos[i+14] = LeftArmOutput.realJointPos[i-6] * M_PI / 180 - default_angle[i+13];
        }
        dof_pos[19] = 0.0;
        dof_pos[26] = 0.0;
        dof_pos[18] = 0.0;
        dof_pos[25] = 0.0;
    }
    return dof_pos;
}

// 3. vel 12d
std::vector<float> HumanoidController::Vel()
{   std::vector<float> dof_vel(27);
    for (int i = 0; i < 13; ++i)
    {
        bridge_->getQDDLegsAndWaistInfo(LWInfo);
        if (i < 6)
        {
            // dof_vel[i] = LWInfo.realJointVel[i] * 0.05f;
            dof_vel[i] = LWInfo.realJointVel[i + 9] * M_PI / 180;
        }
        else if (i < 12)
        {
            // dof_vel[i] = LWInfo.realJointVel[i + 3] * 0.05f;
            dof_vel[i] = LWInfo.realJointVel[i - 6] * M_PI / 180;
        }
        else
        {
            dof_vel[i] = LWInfo.realJointVel[8] * M_PI / 180;
            // dof_pos[i] = 0;
        }
        // std::cout << "Vel: (" << dof_vel[i] << ", " << ")\n";
    }
    for (int i = 0; i < 12; ++i)
    {
        bridge_->getLeftArmInformation(LeftArmOutput);
        bridge_->getRightArmInformation(RightArmOutput);
        if (i < 6)
        {
            dof_vel[i+13] = RightArmOutput.realJointVel[i] * M_PI / 180;
          
        }else{

            dof_vel[i+14] = LeftArmOutput.realJointVel[i-6] * M_PI / 180;
        }
        dof_vel[19] = 0.0;
        dof_vel[26] = 0.0;
        dof_vel[18] = 0.0;
        dof_vel[25] = 0.0;
    }
    return dof_vel;
}

// 4. last action  12d
std::vector<float> HumanoidController::IMUquat()
{
    std::vector<float> quat(4);

    if (bridge_->getImuOutput(imuData_))
    {
        // 从 imuData_.q[] 中获取四元数 (顺序为 [w, x, y, z])
        quat[0] = imuData_.q[0]; //
        quat[1] = imuData_.q[1]; // 虚部 (i)
        quat[2] = imuData_.q[2]; // 虚部 (j)
        quat[3] = imuData_.q[3]; // 虚部 (k)

    }
    else
    {
        std::cerr << "Failed to read IMU data!\n";
    }

    return quat;
}
// 5. 读取 IMU omega 数据  3d
std::vector<float> HumanoidController::IMUomega()
{
    std::vector<float> base_ang_vel(3);
    float ang_vel_scale=1.0f;
    if (bridge_->getImuOutput(imuData_))
    {
        //rad ?
        // base_ang_vel[0] = imuData_.angularSpeed[1];
        // base_ang_vel[1] = -imuData_.angularSpeed[0];
        // base_ang_vel[2] = imuData_.angularSpeed[2];

        base_ang_vel[0] = imuData_.angularSpeed[0] * ang_vel_scale;
        base_ang_vel[1] = imuData_.angularSpeed[1] * ang_vel_scale;
        base_ang_vel[2] = imuData_.angularSpeed[2] * ang_vel_scale;

        // base_ang_vel[0] = -imuData_.angularSpeed[1];
        // base_ang_vel[1] = imuData_.angularSpeed[0];
        // base_ang_vel[2] = imuData_.angularSpeed[2];

        // 打印 IMU 数据（调试用）
        // cout << "Angular Velocity: (" << base_ang_vel[0] << ", " << base_ang_vel[1] << ", " << base_ang_vel[2] << ")\n";
    }
    else
    {
        cerr << "Failed to read IMU data!\n";
    }

    return base_ang_vel;
}

// read real torque
std::vector<float> HumanoidController::Torque()
{
    std::vector<float> dof_torque(25);
    for (int i = 0; i < 13; ++i)
    {
        bridge_->getQDDLegsAndWaistInfo(LWInfo);
        if (i < 6)
        {
            // dof_vel[i] = LWInfo.realJointVel[i] * 0.05f;
            dof_torque[i] = LWInfo.realJointTorque[i + 9];
        }
        else if (i < 12)
        {
            // dof_torque[i] = LWInfo.realJointVel[i + 3] * 0.05f;
            dof_torque[i] = LWInfo.realJointTorque[i - 6];
        }
        else
        {
            dof_torque[i] = LWInfo.realJointTorque[8];
            // dof_pos[i] = 0;
        }
        // std::cout << "Vel: (" << dof_vel[i] << ", " << ")\n";
    }
    for (int i = 0; i < 12; ++i)
    {
        bridge_->getLeftArmInformation(LeftArmOutput);
        bridge_->getRightArmInformation(RightArmOutput);
        if (i < 6)
        {
            dof_torque[i+13] = RightArmOutput.realJointTorque[i];
          
        }else{

            dof_torque[i+13] = LeftArmOutput.realJointTorque[i-6];
        }
    }
    return dof_torque;
}

// read Desired_Iq
std::vector<float> HumanoidController::Desired_Iq()
{
    std::vector<float> Desired_Iq(12);
    for (int i = 0; i < 6; ++i)
    {
        bridge_->getQDDLegsAndWaistOutput(LegsAndWristOutput);
        Desired_Iq[i] = LegsAndWristOutput.rightLeg[i].desiredcurrent;
        Desired_Iq[i+6] = LegsAndWristOutput.leftLeg[i].desiredcurrent;
        // std::cout << "Position: (" << dof_pos[i] << ", " << ")\n";
    }

    return Desired_Iq;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

// 策略更新线程函数
void policyUpdateThread(HumanoidController& controller, torch::jit::script::Module& policy_model) {
     // 初始化timerfd（CLOCK_MONOTONIC：单调时钟，不受系统时间修改影响）
    int timer_fd = timerfd_create(CLOCK_MONOTONIC, 0);
    if (timer_fd == -1) {
        perror("timerfd_create failed");
        return;
    }
    // 配置周期：10ms（目标周期）
    const int period_ms = 20;
    struct itimerspec timer_spec;
    timer_spec.it_interval.tv_sec = 0;                  // 周期的秒部分
    timer_spec.it_interval.tv_nsec = period_ms * 1000000;  // 周期的纳秒部分（10ms = 10,000,000ns）
    timer_spec.it_value = timer_spec.it_interval;        // 首次触发时间与周期相同
    if (timerfd_settime(timer_fd, 0, &timer_spec, nullptr) == -1) {
        perror("timerfd_settime failed");
        close(timer_fd);
        return;
    }
    uint64_t missed = 0;  // 记录错过的周期数
    // 在循环外初始化上一次循环的开始时间
    auto last_start = std::chrono::high_resolution_clock::now();



    std::deque<torch::Tensor> history_inputs;
    for (int i = 0; i < 9; ++i) {
        history_inputs.push_back(torch::zeros({1, 90}));
    }

    while (running) {
        if(!start_model_inference){
            // base_euler_xyz_init = controller.IMUrpy();
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 控制发送频率
        }
        else if(start_model_inference){
            auto current_start = std::chrono::high_resolution_clock::now();  // 本次循环开始时间
            // 计算与上一次循环的时间差（即周期）
            auto cycle = std::chrono::duration_cast<std::chrono::microseconds>(current_start - last_start).count();
            // printf("policy周期: %.3f ms\n", cycle / 1000.0);  // 转换为毫秒
            last_start = current_start;  // 更新上一次开始时间

            // std::cout << "policyUpdateThread\n";
            std::vector<torch::Tensor> inputs;
            // auto loop_start = std::chrono::high_resolution_clock::now();
            // auto loop_end = std::chrono::high_resolution_clock::now();
            // auto total_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start).count();
            // loop_start = loop_end;
            // double total_seconds = total_elapsed / 1e6;
            // printf("Total loop time (after sleep): %.6f s (%.3f ms)\n", total_seconds, total_seconds * 1000);

            // 1. omega 3d
            auto base_ang_vel = controller.IMUomega();
            torch::Tensor base_ang_vel_tensor = torch::from_blob(base_ang_vel.data(), {1, 3});
            inputs.push_back(base_ang_vel_tensor);
            inputs_history.push_back(base_ang_vel);
            

            // 2. gravity_orientation 3d
            auto base_quat = controller.IMUquat();
            std::vector<float> gravity_ori(3);
            gravity_ori[0] = 2*(-base_quat[3]*base_quat[1] + base_quat[0]*base_quat[2]);
            gravity_ori[1] = -2*(base_quat[3]*base_quat[2] + base_quat[0]*base_quat[1]);
            gravity_ori[2] = 1 - 2*(base_quat[0]*base_quat[0] + base_quat[3]*base_quat[3]);
            torch::Tensor gravity_tensor = torch::from_blob(gravity_ori.data(), {1, 3});
            inputs.push_back(gravity_tensor);
            inputs_history.push_back(gravity_ori);


            // 3. cmd 3d
            std::vector<float> commands(3);
            commands[0] = command_x * 1.0f;
            commands[1] = command_y * 1.0f;
            commands[2] = command_yaw * 1.0f;
            torch::Tensor command_tensor = torch::from_blob(commands.data(), {1, 3});
            inputs.push_back(command_tensor);
            inputs_history.push_back(commands);
            
            // 5. qj num_actions
            auto dof_pos = controller.Position();
            std::vector<float> dof_pos_obs(27);
            for(int i=0; i<27; ++i)
            {
                dof_pos_obs[i]=dof_pos[usd2urdf[i]];
            }
            torch::Tensor dof_pos_tensor = torch::from_blob(dof_pos_obs.data(), {1, 27});
            inputs.push_back(dof_pos_tensor);
            inputs_history.push_back(dof_pos);
            
            // 6. dqj num_actions
            auto dof_vel = controller.Vel();
            std::vector<float> dof_vel_obs(27);
            for(int i=0; i<27; ++i)
            {
                dof_vel_obs[i]=dof_vel[usd2urdf[i]];
            }
            torch::Tensor dof_vel_tensor = torch::from_blob(dof_vel_obs.data(), {1, 27});
            inputs.push_back(dof_vel_tensor);
            inputs_history.push_back(dof_vel);

            // 7. action num_actions  we have last_actions to get 
            torch::Tensor last_actions_tensor = torch::from_blob(last_actions.data(), {1, 27});
            inputs.push_back(last_actions_tensor);
            inputs_history.push_back(last_actions);

            // // 8. sin/cos_phase 2d
            // std::vector<float> phase(2);
            // std::vector<float> phi(1);
            // float period=0.7;
            // if(command_stand==true){
            //     phase[0] = 0;
            //     phase[1] = 0;
            // }else{
            //     phase[0] = std::sin(2 * M_PI * count_dt * 0.01 / period);
            //     phase[1] = std::cos(2 * M_PI * count_dt * 0.01 / period);
            // }
            // phi[0] = count_dt * 0.01 / period;
            // torch::Tensor phase_tensor = torch::from_blob(phase.data(), {1, 2});
            // inputs.push_back(phase_tensor);
            // inputs_history.push_back(phase);
            // inputs_history.push_back(phi);

            torch::Tensor inputs_tensor = torch::cat(inputs, 1);
            
            history_inputs.push_back(inputs_tensor);

            if (history_inputs.size() > 10) {
                history_inputs.pop_front();
            }
            // 15*56 --> 1*840
            std::vector<torch::Tensor> inputs_all(history_inputs.begin(), history_inputs.end());
            torch::Tensor inputs_all_tensor = torch::cat(inputs_all, 1);
            // 创建视图
            // torch::Tensor reshaped_view = inputs_all_tensor.view({15, 59});

            // 自定义打印格式
            // std::cout << std::fixed << std::setprecision(4);
            // const int cols_per_line = 10;  // 每行显示10个维度

            // for (int i = 0; i < reshaped_view.size(0); ++i) {
            //     // std::cout << "向量 " << i << ":\n";
    
            //     // 遍历该向量的所有维度
            //     for (int start_col = 0; start_col < reshaped_view.size(1); start_col += cols_per_line) {
            //         int end_col = std::min(static_cast<int64_t>(start_col + cols_per_line), 
            //          reshaped_view.size(1));
        
            //         // // 打印列标题
            //         // std::cout << " 维度 " << (start_col + 1) << "-" << end_col << ": ";
        
            //         // // 打印当前块的维度值
            //         // for (int j = start_col; j < end_col; ++j) {
            //         //     std::cout << reshaped_view[i][j].item<float>();
            //         //     if (j < end_col - 1) std::cout << " ";  // 不是最后一个元素加空格
            //         // }
            //         // std::cout << "\n";
            //     }
            // }

            // // 执行策略网络推理
            // std::cout<<"inputs_all_tensor"<<inputs_all_tensor<<std::endl;
            at::Tensor output;
            try {
                output = policy_model.forward({inputs_all_tensor}).toTensor();
            } catch (const c10::Error &e) {
                std::cerr << "Error during inference: " << e.what() << std::endl;
                continue;
            }
            if(start_control_send){
                count_dt = count_dt + 1;
            }
            // 更新 last_actions
            {

                std::lock_guard<std::mutex> lock(mtx);
                for (int i = 0; i < 27; ++i) {
                    last_actions[i] = output[0][i].item<float>();
                    // last_actions_ang[i] = (default_angle[i]+last_actions[i]*0.5)*180/M_PI  ;
                }
            }
            // std::cout<<"last_actions"<<last_actions_ang<<std::endl;


            // auto loop_end = std::chrono::high_resolution_clock::now();
            // auto total_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start).count();
            // double total_seconds = total_elapsed / 1e6;
            // printf("Total loop time (after sleep): %.6f s (%.3f ms)\n", total_seconds, total_seconds * 1000);


            // cv_cond.notify_one(); // 通知控制线程
            // loop_start = loop_end;
            // std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 控制更新频率


            // 核心：用timerfd等待周期，确保固定频率
            // --------------------------
            // 读取timerfd，阻塞等待下一个周期（若任务执行超时，会立即返回并记录missed）
            ssize_t ret = read(timer_fd, &missed, sizeof(missed));
            if (ret != sizeof(missed)) {
                perror("read timerfd failed");
                break;
            }
        }
    }
    
}

// 控制指令发送线程
void controlSendThread(HumanoidController& controller) {
    // 初始化timerfd（CLOCK_MONOTONIC：单调时钟，不受系统时间修改影响）
    int timer_fd = timerfd_create(CLOCK_MONOTONIC, 0);
    if (timer_fd == -1) {
        perror("timerfd_create failed");
        return;
    }

    // 配置周期：10ms（目标周期）
    const int period_ms = 5;
    struct itimerspec timer_spec;
    timer_spec.it_interval.tv_sec = 0;                  // 周期的秒部分
    timer_spec.it_interval.tv_nsec = period_ms * 1000000;  // 周期的纳秒部分（10ms = 10,000,000ns）
    timer_spec.it_value = timer_spec.it_interval;        // 首次触发时间与周期相同
    if (timerfd_settime(timer_fd, 0, &timer_spec, nullptr) == -1) {
        perror("timerfd_settime failed");
        close(timer_fd);
        return;
    }
    uint64_t missed = 0;  // 记录错过的周期数
    // 在循环外初始化上一次循环的开始时间
    auto last_start = std::chrono::high_resolution_clock::now();


    // float LeftcurAngle[7], RightcurAngle[7], curVel[7], curAcc[7], LeftfeedbackTorque[7], RightfeedbackTorque[7];
    std::vector<float> last_received_actions(27, 0.0f); // 保存上一次接收到的控制指令
    std::vector<float> 
    current_tau(25, 0.0f); // 当前的 tau 值
    std::vector<float> current_dof_torque(25, 0.0f); // 当前的 dof_torque 值
    std::vector<float> current_dof_pos(25, 0.0f);
    std::vector<float> current_dof_vel(25, 0.0f);
    std::vector<float> current_desired_Iq(12, 0.0f);
    cout <<start_model_inference;
    float tau;
    float qd;
    auto loop_start = std::chrono::high_resolution_clock::now();

    while (running) {
        if(start_model_inference && start_control_send){
            auto current_start = std::chrono::high_resolution_clock::now();  // 本次循环开始时间
            // 计算与上一次循环的时间差（即周期）
            auto cycle = std::chrono::duration_cast<std::chrono::microseconds>(current_start - last_start).count();
            // printf("cmd周期: %.3f ms\n", cycle / 1000.0);  // 转换为毫秒
            last_start = current_start;  // 更新上一次开始时间
            

            std::vector<float> actions;
            // 检查是否有新的控制指令
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (!last_actions.empty()) {
                    last_received_actions = last_actions; // 更新为最新的控制指令
                }
                actions = last_received_actions; // 使用最新的或上一次的控制指令
                // action_store = actions;
            }
            // std::cout << "actions: (" << actions[0] << ", " << ")\n";
            auto dof_pos = controller.Position();
            auto dof_vel = controller.Vel();
            current_dof_torque = controller.Torque(); // 获取当前的 dof_torque 值
            // printf("current_dof_torque: ( %.3f)\n");
            for(int j=0; j<25; j++){
                dof_pos[j] = dof_pos[j] + default_angle[j];
                current_dof_pos[j] = dof_pos[j];
            }
            for(int k=0; k<25; k++){
                dof_vel[k] = dof_vel[k] * 20;
                current_dof_vel[k] = dof_vel[k];
            }
            current_desired_Iq = controller.Desired_Iq();
            current_desired_Iq[0] = current_desired_Iq[0] / Iq_torque[0];
            current_desired_Iq[1] = current_desired_Iq[1] / Iq_torque[1];
            current_desired_Iq[2] = -current_desired_Iq[2] / Iq_torque[2];
            current_desired_Iq[3] = current_desired_Iq[3] / Iq_torque[3];
            current_desired_Iq[4] = current_desired_Iq[4] / Iq_torque[4];
            current_desired_Iq[5] = current_desired_Iq[5] / Iq_torque[5];
            current_desired_Iq[6] = -current_desired_Iq[6] / Iq_torque[6];
            current_desired_Iq[7] = current_desired_Iq[7] / Iq_torque[7];
            current_desired_Iq[8] = -current_desired_Iq[8] / Iq_torque[8];
            current_desired_Iq[9] = -current_desired_Iq[9] / Iq_torque[9];
            current_desired_Iq[10] = -current_desired_Iq[10] / Iq_torque[10];
            current_desired_Iq[11] = current_desired_Iq[11] / Iq_torque[11];
            
            // Position Control
            float action_scale=0.25f;
            for (int i = 0; i < 27; i++) 
            {
                target_dof_pos[usd2urdf[i]]=actions[i] * action_scale; 
            }
            for (int i = 0; i < 6; i++) {
                // tau = Kp[i] * (target_dof_pos[i] + default_angle[i] - dof_pos[i]) + Kd[i] * (0 - dof_vel[i]);
                // current_tau[i] = tau; // 记录当前 tau 值
                //  printf("current_tau[i]:, %.6f, \n",current_tau[i]);
                // qd = actions[i] * 0.25f + default_angle[i]; 
                // printf("joint[%d]:, %.6f, \n",i, qd);
                RightcurAngle[i] = (target_dof_pos[i] + default_angle[i]) * 180 / M_PI;
                // std::cout << "Position: (" << LeftcurAngle[i] << ", " << ")\n";
                curVel[i] = 0.0f;
                curAcc[i] = 0.0f;
            }
            for (int i = 0; i < 6; i++) {
                // tau = Kp[i+6] * (actions[i+6] * action_scale + default_angle[i+6] - dof_pos[i+6]) + Kd[i+7] * (0 - dof_vel[i+6]);
                // current_tau[i+6] = tau; // 记录当前 tau 值
                // printf("joint[%d]:, %.6f, \n",i+6, qd);
                LeftcurAngle[i] = (target_dof_pos[i+6] + default_angle[i+6]) * 180 / M_PI;
                curVel[i] = 0.0f;
                curAcc[i] = 0.0f;
            }
            for (int i = 0; i < 5; i++) {
                // tau = Kp[i+6] * (actions[i+6] * action_scale + default_angle[i+6] - dof_pos[i+6]) + Kd[i+7] * (0 - dof_vel[i+6]);
                // current_tau[i+6] = tau; // 记录当前 tau 值
                // printf("joint[%d]:, %.6f, \n",i+6, qd);
                RightcurAngle_arm[i] = (target_dof_pos[i+13] + default_angle[i+13]) * 180 / M_PI;
                curVel[i] = 0.0f;
                curAcc[i] = 0.0f;
            }
            for (int i = 0; i < 5; i++) {
                // tau = Kp[i+6] * (actions[i+6] * action_scale + default_angle[i+6] - dof_pos[i+6]) + Kd[i+7] * (0 - dof_vel[i+6]);
                // current_tau[i+6] = tau; // 记录当前 tau 值
                // printf("joint[%d]:, %.6f, \n",i+6, qd);
                LeftcurAngle_arm[i] = (target_dof_pos[i+20] + default_angle[i+20]) * 180 / M_PI;
                curVel[i] = 0.0f;
                curAcc[i] = 0.0f;
            }
            WaistAngle[2] = (target_dof_pos[12] + default_angle[12]) * 180 / M_PI;
            controller.RTmultidoftrajplanModeWithDynaKpKdTorquecmd(LeftcurAngle_arm, curVel, curAcc, Kp_arm, Kd_arm, curVel,LEFTARM);
            controller.RTmultidoftrajplanModeWithDynaKpKdTorquecmd(RightcurAngle_arm, curVel, curAcc, Kp_arm, Kd_arm, curVel,RIGHTARM);
            controller.RTmultidoftrajplanModeWithDynaKpKdTorquecmd(LeftcurAngle, curVel, curAcc, Kp_walk, Kd_walk, curVel, QDDLEFTLEG);
            controller.RTmultidoftrajplanModeWithDynaKpKdTorquecmd(RightcurAngle, curVel, curAcc, Kp_walk, Kd_walk, curVel, QDDRIGHTLEG);
            controller.RTmultidoftrajplanModeWithDynaKpKdTorquecmd(WaistAngle, curVel, curAcc, Kp_waist, Kd_waist, curVel, QDDWAIST);
            // 将当前 tau 值添加到历史记录
            std::lock_guard<std::mutex> lock_tau(tau_history_mutex);
            // tau_history.push_back(current_tau);
            // 将当前 dof_torque 值添加到历史记录
            std::lock_guard<std::mutex> lock_dof_torque(dof_torque_history_mutex);
            dof_torque_history.push_back(current_dof_torque);
            // 将当前 dof_pos 值添加到历史记录
            std::lock_guard<std::mutex> lock_dof_position(dof_pos_history_mutex);
            dof_pos_history.push_back(current_dof_pos);
            // 将当前 dof_vel 值添加到历史记录
            std::lock_guard<std::mutex> lock_dof_velocity(dof_pos_send_mutex);
            dof_vel_history.push_back(actions);
            // 将当前 current_desired_Iq 值添加到历史记录
            std::lock_guard<std::mutex> lock_current_desired_Iq(desired_Iq_history_mutex);
            desired_Iq_history.push_back(current_desired_Iq);
            
            // count_dt = count_dt + 1;
            // loop_start = loop_end;
            // std::this_thread::sleep_for(std::chrono::milliseconds(2)); // 控制发送频率
            // 核心：用timerfd等待周期，确保固定频率
            // --------------------------
            // 读取timerfd，阻塞等待下一个周期（若任务执行超时，会立即返回并记录missed）
            ssize_t ret = read(timer_fd, &missed, sizeof(missed));
            if (ret != sizeof(missed)) {
                perror("read timerfd failed");
                break;
            }
        }
        else
        {
           std::this_thread::sleep_for(std::chrono::milliseconds(5)); // 控制发送频率
        }
    }

}

std::vector<float> HumanoidController::smooth_joint_action(float ratio, const std::vector<float>& end_joint_act) {
    for (int i = 0; i < 25; i++) {
        joint_act[i] = (1 - ratio) * init_joint_act[i] + ratio * end_joint_act[i];
        joint_act[i] = std::max(act_pos_low[i], std::min(joint_act[i], act_pos_high[i]));
    }
    return joint_act;
}

//  实现 smooth_joint_action 函数#####
std::vector<float> HumanoidController::smoothTransition(float ratio, float target_ratio) {
    // float ratio = 0.0f; // 平滑过渡的比例
    // float target_ratio = 1.0f; // 最终过渡到目标关节角度的比例
    // float increment = 0.00005f; // 每次增加的比例
    std::vector<float> target_joint_act = {-0.35, 0.06, 0.18, 0.72, -0.4, 0.0, -0.35, -0.06, -0.18, 0.72, -0.4, 0.0, 0.0, 0, -0.1, 0, 0, 0.0, 0.0, -0, 0.1, 0, 0, 0.0, 0.0}; // 目标关节角度
    for (int i = 0; i < 25; i++) {
        target_joint_act[i] = (target_joint_act[i]) * 180 / M_PI; 
    }
    if (ratio < target_ratio) {
        // ratio += increment;
        joint_act_ = smooth_joint_action(ratio, target_joint_act);
        // std::cout << joint_act_;
    }
    else {
        is_initial_transition_done = true; // 标记平滑过渡完成
    }
    // std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 控制平滑过渡的频率

    return joint_act_;
}

bool Init_Pos = true;
void Control_SendThread(HumanoidController& controller) {
    std::cout << "Control_SendThread\n";
    float ratio_ = 0.0f;
    float target_ratio_ = 1.0f;
    float increment_ = 0.0002f;
    int a =0;
    if(Init_Pos)
    {
        while(a<100)
        {
        init_joint_act = controller.Init_Position();
            // for (int i = 0; i < 12; i++) 
            // {
            // printf("[%d]:%f\n",i,init_joint_act[i]);   
            // }
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 控制发送频率
        a++;
        }
    }
    Init_Pos = false;
    printf("**************Init_Position**************\n"); 
    //motor send
    while (smooth) {
        if(!start_model_inference){
            // std::cout << is_initial_transition_done;
            // 执行平滑过渡逻辑
            if(!is_initial_transition_done){
                joint_action = controller.smoothTransition(ratio_, target_ratio_);
                ratio_ += increment_;
                // std::cout << "Control_SendThread\n";
            }
            else{
                if(!promptPrinted){
                    for (int i = 0; i < 25; i++) 
                    {
                        printf("joint[%d]:%f\n",i,joint_action[i]);   
                    }
                    std::cout << "Press space to start model inference...\n";
                    promptPrinted = true;
                    smooth = false;
                }
            }
            std::vector<float> actions;

            for (int i = 0; i < 6; i++) {
                RightcurAngle[i] = (joint_action[i]);
                curVel[i] = 0.0f;
                curAcc[i] = 0.0f;
                // printf("left[%d]:%f\n",i,LeftcurAngle[i]);
                if(std::abs(RightcurAngle[i])>=80){
                    printf("right[%d]:%f\n",i,RightcurAngle[i]);
                    std::this_thread::sleep_for(std::chrono::seconds(10));
                }   
            }
            // LeftcurAngle[0] = -21.198; LeftcurAngle[1] = -1.719;LeftcurAngle[2] = -11.459;LeftcurAngle[3] = 36.095;LeftcurAngle[4] = -17.188;LeftcurAngle[5] = 0;
            controller.RTmultidoftrajplanModeWithDynaKpKdTorquecmd(RightcurAngle, curVel, curAcc, Kp_stand, Kd_stand, curVel, QDDRIGHTLEG);
            
            for (int i = 0; i < 6; i++) {
                LeftcurAngle[i] = (joint_action[i + 6]);
                curVel[i] = 0.0f;
                curAcc[i] = 0.0f;
                // printf("right[%d]:%f\n",i+6,RightcurAngle[i]);
                if(std::abs(LeftcurAngle[i])>=80){
                    printf("left[%d]:%f\n",i,LeftcurAngle[i]);
                    std::this_thread::sleep_for(std::chrono::seconds(10));
                }  
            }
            // RightcurAngle[0]=-21.198;RightcurAngle[1]=1.719;RightcurAngle[2]=11.459;RightcurAngle[3]=36.095;RightcurAngle[4]=-17.188;RightcurAngle[5]=0;
            controller.RTmultidoftrajplanModeWithDynaKpKdTorquecmd(LeftcurAngle, curVel, curAcc, Kp_stand, Kd_stand, curVel, QDDLEFTLEG);


            // 左臂
            for (int i = 0; i < 5; i++) {
                RightcurAngle_arm[i] = (joint_action[i+13]);
                curVel_arm[i] = 0.0f;
                curAcc_arm[i] = 0.0f;
                 if(std::abs(RightcurAngle_arm[i])>=80){
                    printf("right_arm[%d]:%f\n",i,RightcurAngle[i]);
                    std::this_thread::sleep_for(std::chrono::seconds(10));
                }    
            }
            
            controller.RTmultidoftrajplanModeWithDynaKpKdTorquecmd(RightcurAngle_arm, curVel_arm, curAcc_arm, Kp_stand, Kd_stand, curVel, RIGHTARM);
            // controller.RTmultidoftrajplanModecmd(LeftcurAngle_arm, curVel_arm, curAcc_arm, LEFTARM);
            // controller.RTmultidoftrajplanModeWithDynaKpKdTorquecmd(LeftcurAngle, curVel, curAcc, kp_stand, kd_stand, Torque_stand_l, QDDLEFTLEG);
            // 右臂
            for (int i = 0; i < 5; i++) {
                LeftcurAngle_arm[i] = (joint_action[i + 19]);
                // std::cout<<"joint_action"<<joint_action[19]<<std::endl;
                // std::cout<<"RightcurAngle_arm"<<RightcurAngle_arm[1]<<std::endl;
                curVel_arm[i] = 0.0f;
                curAcc_arm[i] = 0.0f;
                 if(std::abs(LeftcurAngle_arm[i])>=80){
                    printf("left_arm[%d]:%f\n",i,LeftcurAngle_arm[i]);
                    std::this_thread::sleep_for(std::chrono::seconds(10));
                }  
            }
            // LeftcurAngle_arm[5]=0.5;
            // LeftcurAngle_arm[6]=-1.4;
            controller.RTmultidoftrajplanModeWithDynaKpKdTorquecmd(LeftcurAngle_arm, curVel_arm, curAcc_arm, Kp_stand, Kd_stand, curVel, LEFTARM);
            // controller.RTmultidoftrajplanModecmd(RightcurAngle_arm, curVel_arm, curAcc_arm, RIGHTARM);
            
            // Waist
            WaistAngle[2] = joint_action[12];
            controller.RTmultidoftrajplanModeWithDynaKpKdTorquecmd(WaistAngle, curVel, curAcc, Kp_walk, Kd_walk, curVel, QDDWAIST);
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 控制发送频率
        }
    }

}


////////////////////////////////////////////////////////////////
void keyboardInputThread(HumanoidController& controller) {
    fd_set readfds;
    struct timeval timeout;
    termios oldt, newt;

    // 设置终端属性为非规范模式
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO); // 关闭规范模式和回显
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    std::cout << "Keyboard thread started.\n";
    std::cout << "Press SPACE to start model inference\n";
    std::cout << "Press H to start control mode\n";
    std::cout << "Press Q to quit\n";
    std::cout << "Use WASD, JL for movement control\n";
    while (running) {
        FD_ZERO(&readfds);
        FD_SET(STDIN_FILENO, &readfds);
        timeout.tv_sec = 0; // 设置超时时间为0秒
        timeout.tv_usec = 10000; // 设置超时时间为10毫秒

        int activity = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &timeout);

        if (activity > 0 && FD_ISSET(STDIN_FILENO, &readfds)) {
            char input;
            if (read(STDIN_FILENO, &input, 1) == 1) { // 使用 read 替代 getchar
                if (input == ' ') { // 检测空格键
                    std::cout << "Space pressed. Starting model inference...\n";
                    start_model_inference = true; // 设置标志
                    start_control_send = true;
                    // break; // 退出线程
                }
                if (input == 'h' || input == 'H') { // 检测H键
                    std::cout << "H pressed. Starting control mode...\n";
                    start_control_send = true; // 设置标志
                }
                if (input == 'w' || input == 'W') { // 前进
                   command_x += 0.1;
                   std::cout << "command_x:"<<command_x<<std::endl;
                }if (input == 's' || input == 'S') { // 后退
                   command_x -= 0.1;
                   std::cout << "command_x:"<<command_x<<std::endl;
                }if (input == 'a' || input == 'A') { // 左移
                   command_y += 0.1;
                   std::cout << "command_y:"<<command_y<<std::endl;
                }if (input == 'd' || input == 'D') { // 右移
                   command_y -= 0.1;
                   std::cout << "command_y:"<<command_y<<std::endl;
                }if (input == 'j' || input == 'J') { // 左转
                   command_yaw += 0.1;
                   std::cout << "command_yaw:"<<command_yaw<<std::endl;
                }if (input == 'l' || input == 'L') { // 右转
                   command_yaw -= 0.1;
                   std::cout << "command_yaw:"<<command_yaw<<std::endl;
                }if (input == 'v' || input == 'V') { // 站立
                   command_stand = true;
                   std::cout << "command_stand:"<<command_stand<<std::endl;
                }if (input == 'm' || input == 'M') { // 停止站立
                   command_stand = false;
                   std::cout << "command_stand:"<<command_stand<<std::endl;
                }
                
            }
        } else {
            // 如果没有检测到输入，可以在这里执行其他任务
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // 恢复终端属性
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
}

// void joystickInputThread(HumanoidController& controller) {
//     fd_set readfds;
//     struct timeval timeout;
//     termios oldt, newt;
//     int xbox_fd ;  
//     xbox_map_t map;  
//     int len, type;  
//     int axis_value, button_value;  
//     int number_of_axis, number_of_buttons ;  

//     memset(&map, 0, sizeof(xbox_map_t));  

//     xbox_fd = xbox_open("/dev/input/js0");  
//     if(xbox_fd < 0)  
//     {  
//         return -1;  
//     }  
//     // 设置终端属性为非规范模式
//     tcgetattr(STDIN_FILENO, &oldt);
//     newt = oldt;
//     newt.c_lflag &= ~(ICANON | ECHO); // 关闭规范模式和回显
//     tcsetattr(STDIN_FILENO, TCSANOW, &newt);

//     while (running) {  
//         len = xbox_map_read(xbox_fd, &map);  
//         if (len < 0)  
//         {  
//             usleep(10*1000);  
//             continue;  
//         }  
//         if (map.a == 1) {
//             std::cout << "A pressed. Starting model inference...\n";
//             start_model_inference = true; // 设置标志
//             continue; // 退出线程
//         }
//         if (std::abs(map.ly)>3300){ // 前进
//             command_x = -0.6*map.ly/32767;
//         }if (std::abs(map.lx)>3300) { // 后退
//             command_y = 0.6*map.lx/32767;
//         }if (std::abs(map.rx)>10000||abs(map.ry)>10000) { // 左转
//             // int sign = (map.rx>0) - (map.rx<0);
//         // 计算角度
//             command_yaw = 0.6*atan2(map.rx, -map.ry)/M_PI;
//         }if (map.lo == 1) { // 站立
//             command_stand = true;
//         }if (map.ro == 1) { // 停止站立
//             command_stand = false;
//         }if (map.back == 1) { // 停止站立
//             break;
//         }
//         printf("\rTime:%8d Start:%d command_x:%f command_y:%f command_yaw:%f command_stand:%d",  
//                 map.time, start_model_inference, command_x, command_y, command_yaw, command_stand);  
//         fflush(stdout);      
//     }
//     // 恢复终端属性
//     tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
// }

// 信号处理函数
void handle_interrupt(int sig) {
    // 设置标志位，表示程序需要安全退出
    running = false;
    std::cout << "Caught Ctrl+C. Exiting gracefully...\n";
    // 调用保存 tau 值的函数
    save_data_to_csv();
    exit(sig); // 退出程序
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "bridge");
    ros::NodeHandle h;
    Bridge bridge(h);
    if (!bridge.Init()) {
        printf("bridge init failed\n");
    } else {
        printf("node running\n");
    }

    // ros::Rate loop_rate(200);
    // Interface::ArmCommand test = {0};
    // while (true) {
    // test.paraLen++;
    // bridge.getQDDLegsAndWaistCommand(test);
    // loop_rate.sleep();
    // }
  
    // add
    // 注册信号处理函数
    signal(SIGINT, handle_interrupt);
    HumanoidController controller(&bridge);
    controller.InitAll();
    // 加载策略模型
    torch::jit::script::Module policy_model;
    try {
        policy_model = torch::jit::load("/home/gy/slrBridge/src/policy_syb/policy.pt");
        std::cout << "Policy model loaded successfully.\n";
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the policy model\n";
        return 0;
    }
    
    policy_model.eval(); 
    auto dummy=torch::zeros({1,900});
    for(int i=0;i<20;i++){
        at::Tensor output;
        try {
            output = policy_model.forward({dummy}).toTensor();
        } catch (const c10::Error &e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            continue;
        }
    }


    // 启动策略更新线程
    std::thread policyThread(policyUpdateThread, std::ref(controller), std::ref(policy_model));
    std::thread controlThread(controlSendThread, std::ref(controller));
    std::thread Control_Thread(Control_SendThread, std::ref(controller));
    std::thread keyboardThread(keyboardInputThread, std::ref(controller));
    // std::thread joystickThread(joystickInputThread, std::ref(controller));
    // std::thread Control_Thread(&HumanoidController::Control_SendThread, &controller);
    // // 启动键盘输入检测线程
    // std::thread keyboardThread(&HumanoidController::keyboardInputThread, &controller);
    
     // ========== 设置线程优先级 ==========
    // 定义调度参数结构体
    // struct sched_param param;
    // int policy = SCHED_FIFO;  // 选择调度策略，也可以用 SCHED_RR
    // int cmd = SCHED_FIFO;
    // 设置 policyThread 的优先级
    // param.sched_priority = 50;  // 优先级值（范围取决于系统，通常 1~99 为实时优先级）
    // if (pthread_setschedparam(policyThread.native_handle(), policy, &param) != 0) {
    //     std::cerr << "Failed to set policyThread priority\n";
    // }

    // // // 设置 controlThread 的优先级
    // param.sched_priority = 99;
    // if (pthread_setschedparam(controlThread.native_handle(), cmd, &param) != 0) {
    //     std::cerr << "Failed to set controlThread priority\n";
    // }

    ros::AsyncSpinner spinner(1);
    spinner.start();
    
    printf("All threads started. Waiting for ROS shutdown...\n");
    
    // 等待ROS关闭信号
    ros::waitForShutdown();
    
    // ========== 清理代码 ==========
    printf("ROS shutdown signal received. Stopping threads...\n");
    running = false;
    // 主线程等待
    policyThread.join();
    controlThread.join();
    Control_Thread.join();
    keyboardThread.join();
    // joystickThread.join();
    
    return 0;

}