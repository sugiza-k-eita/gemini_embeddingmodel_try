import streamlit as st
import os
import io
from dotenv import load_dotenv
from PIL import Image
import numpy as np

# --- 公式ドキュメント準拠：最新SDKのみを使用 ---
from google import genai
from google.genai import types

# --- .env の読み込み ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- ページ設定 ---
st.set_page_config(page_title="Gemini Embedding v2 vs v1 比較", layout="wide")
st.title("🚀 Google Embedding Model 比較ハンズオン")

# ★いただいたフォーラムの情報を検証記事のTipsとしてUIに表示
st.info("""
**💡 TIPS: 旧型モデルの廃止と移行について** 2026年2月に、旧型の `text-embedding-004` や `embedding-001` は古いアーキテクチャとして公式に廃止（404エラー）されました。
そのため、本検証ではGoogle公式フォーラムでの推奨に従い、現在のテキスト専用の安定版モデルである **`gemini-embedding-001`** を旧型パイプラインのベースラインとして使用します。
（※アーキテクチャが刷新されたため、旧モデルで作成したベクトルDBは互換性がなく、再ベクトル化が必須となります）
""")

# --- APIキーの確認 ---
if not api_key:
    st.error("エラー: .env ファイルが見つからないか、GEMINI_API_KEY が設定されていません。")
    st.stop()

# --- クライアントの初期化 ---
client = genai.Client(api_key=api_key)

# --- サイドバー：設定 ---
st.sidebar.header("設定")

model_option = st.sidebar.selectbox(
    "使用する比較パターン",
    ("新型パイプライン (gemini-embedding-2-preview)", "旧型パイプライン (Captioning + gemini-embedding-001)")
)

output_dimensionality = 3072  # Default
if model_option == "新型パイプライン (gemini-embedding-2-preview)":
    output_dimensionality = st.sidebar.selectbox(
        "出力次元数 (MRL)",
        (3072, 1536, 768, 512, 256, 128),
        index=0
    )

# ==========================================
# 【新型パイプライン】 gemini-embedding-2-preview
# ==========================================
def get_embedding_v2(content, output_dim):
    try:
        if isinstance(content, str):
            contents = content
        else:
            img_byte_arr = io.BytesIO()
            content.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            contents = [
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type='image/jpeg',
                )
            ]

        response = client.models.embed_content(
            model='gemini-embedding-2-preview',
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=output_dim)
        )
        return response.embeddings[0].values
    except Exception as e:
        st.error(f"新型Embedding取得エラー: {e}")
        return None

# ==========================================
# 【旧型パイプライン】 画像の言語化 → テキスト専用Embedding
# ==========================================
def generate_caption(image):
    try:
        prompt = "この画像の具体的な特徴（色、素材、物体、雰囲気）を、検索クエリで使われそうな短い英語の単語やフレーズでリストアップしてください。"
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image]
        )
        return response.text
    except Exception as e:
        st.error(f"キャプション生成エラー: {e}")
        return None

def get_embedding_v1(text):
    """
    ★フォーラムの解決策通り、新しいエンドポイントの gemini-embedding-001 を呼び出します。
    """
    try:
        response = client.models.embed_content(
            model='gemini-embedding-001',
            contents=text
        )
        return response.embeddings[0].values
    except Exception as e:
        st.error(f"旧型Embedding取得エラー: {e}")
        return None

# ==========================================
# 共通のユーティリティ関数
# ==========================================
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# --- メインエリア ---

st.header("Step 1: 検証用画像の準備")
st.markdown("画面からアップロードするか、`images/` フォルダに画像を配置してください。")

# 1. 画面からのアップロード
uploaded_files = st.file_uploader("画像をアップロード...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# 2. ローカルフォルダからの自動読み込みロジック
local_images_dir = "./images"
local_image_paths = []
if os.path.exists(local_images_dir):
    for f in os.listdir(local_images_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            local_image_paths.append(os.path.join(local_images_dir, f))

uploaded_images = []

# 画像のセットアップ
if uploaded_files:
    for file in uploaded_files:
        uploaded_images.append(Image.open(file))
elif local_image_paths:
    st.info(f"📁 `images/` フォルダから {len(local_image_paths)} 枚の画像を自動検出しました。")
    for path in local_image_paths:
        uploaded_images.append(Image.open(path))

# 画像のプレビュー表示
if uploaded_images:
    cols = st.columns(len(uploaded_images))
    for i, image in enumerate(uploaded_images):
        with cols[i]:
            st.image(image, use_container_width=True, caption=f"Image {i+1}")

    # Step 2: ベクトル化
    st.header("Step 2: 画像をベクトル化してDBに保存")
    if st.button("画像をベクトル化"):
        with st.spinner("画像を処理中... (旧型の場合はキャプション生成に数十秒かかります)"):
            image_vectors = []
            image_captions = []
            
            for i, image in enumerate(uploaded_images):
                if model_option == "新型パイプライン (gemini-embedding-2-preview)":
                    vec = get_embedding_v2(image, output_dimensionality)
                    if vec is not None:
                        image_vectors.append(vec)
                        image_captions.append("(Gemini Embedding v2 directly embedded)")
                    else:
                        image_vectors.append(None)
                else:
                    caption = generate_caption(image)
                    if caption:
                        image_captions.append(caption)
                        vec = get_embedding_v1(caption)
                        if vec is not None:
                            image_vectors.append(vec)
                        else:
                            image_vectors.append(None)
                    else:
                        image_vectors.append(None)
            
            st.session_state['image_vectors'] = image_vectors
            st.session_state['image_captions'] = image_captions
            st.session_state['processed_images_count'] = len([v for v in image_vectors if v is not None])
            st.session_state['last_model'] = model_option
            st.success(f"{st.session_state['processed_images_count']} 枚の画像のベクトル化が完了しました！")

            if "旧型パイプライン" in model_option and image_captions:
                st.subheader("生成されたキャプション (旧型での検索対象)")
                for i, cap in enumerate(image_captions):
                    with st.expander(f"Image {i+1} のキャプション"):
                        st.text(cap.strip() if cap else "生成失敗")


# Step 3: 検索
if 'image_vectors' in st.session_state and st.session_state.get('processed_images_count', 0) > 0:
    st.header("Step 3: テキストクエリで画像を検索")
    
    if st.session_state.get('last_model') != model_option:
        st.warning(f"注意: 現在のモデル設定 ({model_option}) は、保存されたベクトル ({st.session_state.get('last_model')}) と異なります。再度「画像をベクトル化」を実行してください。")
        st.stop()

    search_query = st.text_input("検索クエリ (テキスト) を入力してください", placeholder="例: 木の机、ラテアート")
    
    # ★スライダーのデフォルトを 1 に変更
    top_k = st.slider("表示する上位件数 (Top K)", min_value=1, max_value=len(uploaded_images), value=1)

    if st.button("検索"):
        if not search_query:
            st.warning("クエリを入力してください。")
        else:
            with st.spinner("検索中..."):
                if model_option == "新型パイプライン (gemini-embedding-2-preview)":
                    query_vec = get_embedding_v2(search_query, output_dimensionality)
                else:
                    query_vec = get_embedding_v1(search_query)

                if query_vec is not None:
                    similarities = []
                    for i, img_vec in enumerate(st.session_state['image_vectors']):
                        if img_vec is not None:
                            score = cosine_similarity(query_vec, img_vec)
                            similarities.append((i, score))
                        else:
                            similarities.append((i, -1.0))
                    
                    # 類似度が高い順（降順）にソート
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # 上位 K 件だけをスライスして取得
                    top_results = similarities[:top_k]
                    
                    st.subheader(f"「{search_query}」の検索結果 (Top {top_k})")
                    
                    # ★UI改善: Top1の時はカラム分けせずに大きく表示、2以上の時はカラムで並べる
                    if top_k == 1:
                        rank = 0
                        idx, score = top_results[0]
                        if score != -1.0:
                            # レイアウトを整えるために中央のカラムに配置
                            _, col2, _ = st.columns([1, 2, 1])
                            with col2:
                                st.markdown("### 🏆 Rank 1 (Best Match)")
                                st.image(uploaded_images[idx], use_container_width=True, caption=f"Score: {score:.4f}")
                                if "旧型パイプライン" in model_option:
                                    with st.expander("キャプションを確認"):
                                        st.text(st.session_state['image_captions'][idx].strip())
                    else:
                        cols = st.columns(len(top_results))
                        for rank, (idx, score) in enumerate(top_results):
                            if score != -1.0:
                                with cols[rank]:
                                    st.markdown(f"### 🏆 Rank {rank+1}")
                                    st.image(uploaded_images[idx], use_container_width=True, caption=f"Score: {score:.4f}")
                                    if "旧型パイプライン" in model_option:
                                        with st.popover("キャプションを確認"):
                                            st.text(st.session_state['image_captions'][idx].strip())
