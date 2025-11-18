import os, glob, re
from datetime import datetime
import pandas as pd
import multiprocessing as mp
from ActionLoggerCallbak import ActionLoggerCallback

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from policy import CustomActorCriticPolicy
from trade_env import TradeEngineEnv
from vec_wrappers import LoggingVecNormalize

N_DAYS = 19
VEC_MODE = "subproc"   # "dummy" | "subproc"
PASSES = 100        # quantas passagens sobre os dados
THREADS_PER_PROC = 1  # limita threads BLAS/torch por processo

# Evita over-subscription de threads nos subprocessos/BLAS
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS_PER_PROC))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS_PER_PROC))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREADS_PER_PROC))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(THREADS_PER_PROC))

def extract_date_from_name(path):
    m = re.search(r"(\d{8})", os.path.basename(path))
    return datetime.strptime(m.group(1), "%d%m%Y") if m else None

def make_env(csv_path, *, env_kwargs: dict | None = None, monitor_kwargs: dict | None = None):
    env_kwargs = dict(env_kwargs or {})
    monitor_kwargs = dict(monitor_kwargs or {})
    def _thunk():
        df = pd.read_csv(csv_path)
        default_kwargs = {
            "max_steps": len(df) - 1,
            "log_name": os.path.basename(csv_path),
        }
        default_kwargs.update(env_kwargs)
        env = TradeEngineEnv(df, **default_kwargs)
        env = Monitor(env, **monitor_kwargs)
        return env
    return _thunk

def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), "train_data")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")), key=extract_date_from_name)
    assert files, f"Nenhum arquivo encontrado em {DATA_DIR}"

    n_days = N_DAYS
    selected_files = files[-n_days:] if n_days and n_days > 0 else files
    print(f"{len(files)} arquivos encontrados. Usando {len(selected_files)} arquivos de treino.")

    day_lengths = [len(pd.read_csv(f)) for f in selected_files]
    episode_len = min(day_lengths) - 1
    print(f"Episódio mínimo ~{episode_len} passos")

    env_fns = [make_env(p) for p in selected_files]
    # Decide backend do vetor
    if VEC_MODE == "dummy":
        vec_backend = "dummy"
        vec_env = DummyVecEnv(env_fns)
    if VEC_MODE == "subproc":
        vec_backend = "subproc"
        vec_env = SubprocVecEnv(env_fns, start_method="fork")
    print(f"VecEnv: {vec_backend} | n_envs={len(env_fns)} | cores={mp.cpu_count()}")

    vec_env = LoggingVecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
    )

    # Seleciona dispositivo (GPU se disponível) após criar os subprocessos do env
    import torch
    try:
        torch.set_num_threads(THREADS_PER_PROC)
    except Exception:
        pass
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "cuda"
        print(f"Usando GPU: {gpu_name}")
    else:
        print("CUDA não disponível, treinando em CPU.")

    model = PPO(
        CustomActorCriticPolicy,
        vec_env,
        policy_kwargs={'hidden_dim': (1, 64)},
        n_steps=min(2048, episode_len),
        batch_size=256,
        n_epochs=2,
        ent_coef=0.02,
        learning_rate=3e-4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log="./tb_logs",
        device=DEVICE,
    )

    avg_steps = int(sum(l - 1 for l in day_lengths) / len(day_lengths))
    passes = int(PASSES)
    total_ts = avg_steps * len(selected_files) * passes
    print(f"Treinando por ~{total_ts:,} timesteps "
          f"(avg_steps={avg_steps}, days={len(selected_files)}, passes={passes})")

    model.learn(total_timesteps=total_ts, callback=ActionLoggerCallback())
    model.save("MM_model_trained.zip")
    vec_env.save("vecnormalize_stats.pkl")
    vec_env.close()
    print("Modelo salvo em MM_model_trained.zip")

if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()
