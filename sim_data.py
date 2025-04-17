import numpy as np
import pandas as pd
from scipy.special import expit


class TwoStageTask:

    def __init__(self, task_type=0, seed=None):
        self.rng = np.random.RandomState(seed)
        self.trans_prob = np.array([
            [0, 0.7, 0.3],  
            [0, 0.3, 0.7]
        ])
        self.common = task_type

    def step(self, action):
        """第一阶段"""
        s2 = self.rng.choice([1, 2], p=self.trans_prob[action, 1:])
        return s2, 0, False, {'common': self.common}

    def final_step(self, action, state):
        """第二阶段"""
        if state == 1:
            reward = 1 if (action == 0 and self.common == 0) or (action == 1 and self.common == 1) else 0
        else:
            reward = 1 if (action == 0 and self.common == 1) or (action == 1 and self.common == 0) else 0
        return 0, reward, True, {}

class ParameterGenerator:
    def __init__(self, seed=2025):
        self.rng = np.random.RandomState(seed)
    
    def generate(self, n=100):
        params_list = []
        for _ in range(n):
            params = {
                'beta1': self.rng.uniform(np.log(1), np.log(10)),
                'beta2': self.rng.uniform(np.log(1), np.log(10)),
                'alpha1': self.rng.uniform(-3, 3),
                'alpha2': self.rng.uniform(-3, 3),
                'lmbda': self.rng.uniform(-2, 2),
                'p': self.rng.uniform(0, 2),
                'w': self.rng.uniform(-3, 3)
            }
            params_list.append(params)
        return params_list

class HybridAgent:
    def __init__(self, nS, nA, rng, params):
        self.nS = nS
        self.nA = nA
        self.rng = rng
        self.beta1 = np.exp(params['beta1'])
        self.beta2 = np.exp(params['beta2'])
        self.alpha1 = expit(params['alpha1'])
        self.alpha2 = expit(params['alpha2'])
        self.lmbda = expit(params['lmbda'])
        self.p = params['p']
        self.w = expit(params['w'])
        self.Q_mf = np.zeros((nS, nA))
        self.Q_mb = np.zeros((nS, nA))
        self.trans_matrix = np.array([[0, 0.7, 0.3], [0, 0.3, 0.7]])
        self.rep_a = np.zeros(nA)

    def choose_action(self, state):
        q_net = self.w * self.Q_mb[state] + (1 - self.w) * self.Q_mf[state]
        beta = self.beta1 if state == 0 else self.beta2
        modulated_q = q_net + self.p * self.rep_a
        probs = np.exp(beta * modulated_q)
        probs /= probs.sum()
        return self.rng.choice(self.nA, p=probs)

    def update(self, s1, a1, s2, a2, reward):
        delta2 = reward - self.Q_mf[s2, a2]
        delta1 = self.Q_mf[s2, a2] - self.Q_mf[s1, a1]
        self.Q_mf[s2, a2] += self.alpha2 * delta2
        self.Q_mf[s1, a1] += self.alpha1 * (delta1 + self.lmbda * delta2)
        self.Q_mb[s2, a2] += self.alpha2 * delta2
        self.Q_mb[s1] = self.trans_matrix @ np.max(self.Q_mb, axis=1)
        self.rep_a = np.eye(self.nA)[a1]


def generate_subject_data(params, subj_id, n_trials=500):

    rng = np.random.RandomState(subj_id)
    env = TwoStageTask(task_type=0, seed= 2000 + subj_id)
    agent = HybridAgent(nS=3, nA=2, rng=rng, params=params)

    data = []
    for trial in range(n_trials):
        s1 = 0
        a1 = agent.choose_action(s1)
        s2, _, _, info = env.step(a1)
        a2 = agent.choose_action(s2)
        _, reward, _, _ = env.final_step(a2, s2)
        agent.update(s1, a1, s2, a2, reward)
        
        data.append({
            'trial': trial + 1,
            'act0': a1,
            'state1': s2,
            'act1': a2,
            'reward': reward,
            'block_type': info['common']
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 添加被试ID
    df['subj_id'] = subj_id
    
    # 添加真实参数（转换后）
    true_params = {
        'beta1': np.exp(params['beta1']),
        'beta2': np.exp(params['beta2']),
        'alpha1': expit(params['alpha1']),
        'alpha2': expit(params['alpha2']),
        'lambda': expit(params['lmbda']),
        'p': params['p'],
        'w': expit(params['w'])
    }
    for k, v in true_params.items():
        df[k] = v
    
    # 保存为Excel
    filename = f'param_rev/datum/subj{subj_id}.xlsx'
    df.to_excel(filename, index=False)
    return true_params

# 主程序
if __name__ == "__main__":
    # 生成参数
    param_gen = ParameterGenerator()
    all_raw_params = param_gen.generate(100)
    
    # 存储所有真实参数
    all_true_params = []
    
    # 逐个生成被试数据
    for subj_id, raw_params in enumerate(all_raw_params):
        true_params = generate_subject_data(raw_params, subj_id)
        true_params['subj_id'] = subj_id  # 添加被试ID
        all_true_params.append(true_params)
    
    # 保存真实参数总表
    pd.DataFrame(all_true_params).to_csv('param_rev/true_parameters.csv', index=False)
    
    print("数据生成完成！")
    print("参数总表保存为: true_parameters.csv")
