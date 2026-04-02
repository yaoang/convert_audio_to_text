# 导入所需的库
import os  # 用于操作系统相关功能，如文件和目录操作
import shutil  # 用于高级文件操作，如复制和移动文件
import warnings  # 用于警告控制
import subprocess  # 用于创建子进程
import tempfile  # 用于创建临时文件和目录
import numpy as np  # 用于科学计算和多维数组操作
import torch  # 用于深度学习框架
import soundfile as sf  # 用于音频文件读写
import whisper  # 用于语音识别模型
from speechbrain.inference.speaker import SpeakerRecognition  # 用于说话人识别
from sklearn.cluster import AgglomerativeClustering  # 用于层次聚类
from sklearn.metrics import silhouette_score  # 用于计算轮廓系数

# 忽略所有警告
warnings.filterwarnings('ignore')



# 定义常量
INPUT_DIR = 'input'  # 输入目录路径
OUTPUT_DIR = 'output'  # 输出目录路径
DONE_DIR = 'done'

WHISPER_MODEL = 'base'
NUM_SPEAKERS = None  # Set to an int to fix speaker count, or None to auto-detect (2~8)

SUPPORTED_EXTENSIONS = {'.mp3', '.mp4', '.wav', '.m4a', '.ogg', '.flac', '.webm'}

for d in [INPUT_DIR, OUTPUT_DIR, DONE_DIR]:
    os.makedirs(d, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR)
         if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

if not files:
    print('No audio files found in input/')
    exit(0)

print(f'Found {len(files)} file(s) to process.\n')

print('Loading Whisper model...')
asr_model = whisper.load_model(WHISPER_MODEL)

print('Loading speaker embedding model...')
spk_model = SpeakerRecognition.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='pretrained_models/spkrec-ecapa-voxceleb',
    run_opts={'device': 'cpu'},
)


def load_audio_mono_16k(path):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', '16000', tmp_path],
            check=True, capture_output=True,
        )
        data, _ = sf.read(tmp_path, dtype='float32')
    finally:
        os.remove(tmp_path)
    return torch.from_numpy(data)  # shape: (samples,)


def get_segment_embedding(waveform, start_sec, end_sec, sample_rate=16000):
    start = int(start_sec * sample_rate)
    end = int(end_sec * sample_rate)
    segment = waveform[start:end].unsqueeze(0)  # (1, samples)
    with torch.no_grad():
        embedding = spk_model.encode_batch(segment)
    return embedding.squeeze().numpy()


def cluster_speakers(embeddings, num_speakers):
    if num_speakers is None:
        best_score, best_labels = -1, None
        for k in range(2, min(9, len(embeddings) + 1)):
            if len(embeddings) < k:
                break
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score, best_labels = score, labels
        return best_labels if best_labels is not None else [0] * len(embeddings)
    else:
        return AgglomerativeClustering(n_clusters=num_speakers).fit_predict(embeddings)


def process_file(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    name_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f'{name_without_ext}.txt')
    done_path = os.path.join(DONE_DIR, filename)

    print(f'\nProcessing: {filename}')

    print('  Transcribing...')
    result = asr_model.transcribe(input_path, word_timestamps=False)
    segments = result['segments']

    if not segments:
        print('  No speech detected.')
        return

    print('  Extracting speaker embeddings...')
    waveform = load_audio_mono_16k(input_path)
    embeddings = []
    valid_segments = []
    for seg in segments:
        if seg['end'] - seg['start'] < 0.5:
            continue
        emb = get_segment_embedding(waveform, seg['start'], seg['end'])
        embeddings.append(emb)
        valid_segments.append(seg)

    if not embeddings:
        print('  No valid segments for diarization.')
        return

    print('  Clustering speakers...')
    labels = cluster_speakers(np.array(embeddings), NUM_SPEAKERS)

    # Merge consecutive segments from the same speaker
    merged = []
    for seg, label in zip(valid_segments, labels):
        if merged and merged[-1][2] == label:
            merged[-1][1] = seg['end']
            merged[-1][3] += ' ' + seg['text'].strip()
        else:
            merged.append([seg['start'], seg['end'], label, seg['text'].strip()])

    lines = []
    for start, end, label, text in merged:
        lines.append(f'[{start:.1f}s - {end:.1f}s] Speaker {label + 1}: {text}')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  -> Saved: {name_without_ext}.txt')

    shutil.move(input_path, done_path)
    print(f'  -> Moved to done: {filename}')


for filename in files:
    try:
        process_file(filename)
    except Exception as e:
        print(f'  ERROR processing {filename}: {e}')

print('\nDone.')
