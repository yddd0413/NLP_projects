import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec, FastText
import gensim.downloader as api
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

st.set_page_config(page_title="NLP文本表示工具", layout="wide")

tab1, tab2, tab3, tab4 = st.tabs([
    "模块1: TF-IDF与LSA", 
    "模块2: Word2Vec训练", 
    "模块3: GloVe与词类比", 
    "模块4: FastText与Sent2Vec"
])

with tab1:
    st.header("基于传统统计的文本表示")
    
    default_corpus = """Natural language processing is a subfield of linguistics and artificial intelligence. 
It focuses on the interactions between computers and human language. 
Machine learning algorithms are used to process and analyze text data.
Deep learning has revolutionized the field of natural language processing.
Word embeddings capture semantic relationships between words."""
    
    corpus_text = st.text_area("请输入英文语料:", value=default_corpus, height=200, key="corpus_tab1")
    
    if st.button("分析文本", key="tab1_btn"):
        sentences = sent_tokenize(corpus_text)
        st.subheader(f"文档集合 (共 {len(sentences)} 个句子)")
        for i, sent in enumerate(sentences):
            st.write(f"文档{i+1}: {sent}")
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        st.subheader("TF-IDF矩阵形状")
        st.write(f"形状: {tfidf_matrix.shape} (文档数 x 词汇数)")
        
        st.subheader("TF-IDF权重最高的5个关键词")
        tfidf_sum = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = tfidf_sum.argsort()[-5:][::-1]
        top_keywords = [(feature_names[i], tfidf_sum[i]) for i in top_indices]
        for word, score in top_keywords:
            st.write(f"**{word}**: {score:.4f}")
        
        st.subheader("LSA词汇降维可视化 (2D)")
        count_vec = CountVectorizer()
        count_matrix = count_vec.fit_transform(sentences)
        vocab = count_vec.get_feature_names_out()
        
        svd = TruncatedSVD(n_components=2, random_state=42)
        svd_matrix = svd.fit_transform(count_matrix.T)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(svd_matrix[:, 0], svd_matrix[:, 1], alpha=0.7)
        for i, word in enumerate(vocab):
            ax.annotate(word, (svd_matrix[i, 0], svd_matrix[i, 1]), fontsize=8)
        ax.set_xlabel('维度1')
        ax.set_ylabel('维度2')
        ax.set_title('LSA词汇降维可视化')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

with tab2:
    st.header("Word2Vec实时训练与测试")
    
    w2v_corpus = st.text_area("请输入训练语料:", value=default_corpus, height=150, key="corpus_tab2")
    
    col1, col2 = st.columns(2)
    with col1:
        sg_choice = st.radio("训练架构:", ["CBOW (sg=0)", "Skip-Gram (sg=1)"])
        sg = 0 if "CBOW" in sg_choice else 1
    with col2:
        window_size = st.slider("窗口大小:", min_value=2, max_value=10, value=5)
    
    vector_size = st.slider("向量维度:", min_value=50, max_value=200, value=100, step=50)
    
    w2v_model = None
    
    if st.button("训练Word2Vec模型", key="tab2_btn"):
        sentences_w2v = [word_tokenize(sent.lower()) for sent in sent_tokenize(w2v_corpus)]
        
        with st.spinner("训练中..."):
            w2v_model = Word2Vec(sentences_w2v, vector_size=vector_size, window=window_size, 
                                 sg=sg, min_count=1, workers=4, epochs=100)
        st.success("训练完成!")
        st.session_state['w2v_model'] = w2v_model
        st.session_state['w2v_vocab'] = list(w2v_model.wv.index_to_key)
        st.write(f"词汇表大小: {len(st.session_state['w2v_vocab'])}")
        st.write(f"词汇: {st.session_state['w2v_vocab'][:10]}...")
    
    if 'w2v_model' in st.session_state:
        st.subheader("词相似度测试")
        test_word = st.text_input("输入单词查询相似词:", key="w2v_test_word")
        
        if test_word and st.button("查询相似词", key="query_similar"):
            model = st.session_state['w2v_model']
            if test_word.lower() in model.wv:
                similar_words = model.wv.most_similar(test_word.lower(), topn=5)
                st.write(f"与 '{test_word}' 最相似的5个词:")
                for word, score in similar_words:
                    st.write(f"**{word}**: {score:.4f}")
            else:
                st.warning(f"'{test_word}' 不在词汇表中")

with tab3:
    st.header("预训练GloVe模型与词类比")
    
    @st.cache_resource
    def load_glove_model():
        with st.spinner("加载GloVe模型中..."):
            return api.load("glove-twitter-25")
    
    if 'glove_model' not in st.session_state:
        with st.spinner("首次加载预训练模型，请稍候..."):
            st.session_state['glove_model'] = load_glove_model()
        st.success("模型加载完成!")
    
    glove_model = st.session_state['glove_model']
    
    st.subheader("词类比计算器")
    st.write("计算公式: Result = Vector(A) - Vector(B) + Vector(C)")
    st.write("示例: king - man + woman = queen")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        word_a = st.text_input("词A:", value="king", key="word_a")
    with col2:
        word_b = st.text_input("词B:", value="man", key="word_b")
    with col3:
        word_c = st.text_input("词C:", value="woman", key="word_c")
    
    if st.button("计算词类比", key="analogy_btn"):
        try:
            result = glove_model.most_similar(
                positive=[word_a, word_c],
                negative=[word_b],
                topn=5
            )
            st.write(f"**{word_a} - {word_b} + {word_c}** 的结果:")
            for word, score in result:
                st.write(f"**{word}**: {score:.4f}")
        except KeyError as e:
            st.error(f"词汇不在模型中: {e}")
    
    st.subheader("词义相似度计算")
    col1, col2 = st.columns(2)
    with col1:
        sim_word1 = st.text_input("单词1:", value="computer", key="sim_word1")
    with col2:
        sim_word2 = st.text_input("单词2:", value="laptop", key="sim_word2")
    
    if st.button("计算相似度", key="sim_btn"):
        try:
            similarity = glove_model.similarity(sim_word1, sim_word2)
            st.write(f"'{sim_word1}' 与 '{sim_word2}' 的相似度: **{similarity:.4f}**")
        except KeyError as e:
            st.error(f"词汇不在模型中: {e}")

with tab4:
    st.header("FastText与句子级表示")
    
    ft_corpus = st.text_area("请输入训练语料:", value=default_corpus, height=150, key="corpus_tab4")
    
    if st.button("训练FastText模型", key="ft_train_btn"):
        sentences_ft = [word_tokenize(sent.lower()) for sent in sent_tokenize(ft_corpus)]
        
        with st.spinner("训练FastText模型中..."):
            ft_model = FastText(sentences_ft, vector_size=100, window=5, min_count=1, 
                               workers=4, epochs=100)
        st.success("FastText训练完成!")
        st.session_state['ft_model'] = ft_model
        st.write(f"词汇表大小: {len(ft_model.wv.index_to_key)}")
    
    if 'ft_model' in st.session_state:
        st.subheader("OOV测试 (未登录词)")
        oov_word = st.text_input("输入拼写错误的词进行测试:", value="computeer", key="oov_word")
        
        if st.button("测试OOV处理", key="oov_btn"):
            w2v_model = st.session_state.get('w2v_model')
            ft_model = st.session_state['ft_model']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Word2Vec结果:**")
                if w2v_model:
                    try:
                        vec = w2v_model.wv[oov_word.lower()]
                        similar = w2v_model.wv.most_similar(oov_word.lower(), topn=5)
                        for word, score in similar:
                            st.write(f"{word}: {score:.4f}")
                    except KeyError:
                        st.warning("未登录词 (KeyError)")
                else:
                    st.warning("请先在模块2训练Word2Vec模型")
            
            with col2:
                st.write("**FastText结果:**")
                try:
                    similar_ft = ft_model.wv.most_similar(oov_word.lower(), topn=5)
                    for word, score in similar_ft:
                        st.write(f"{word}: {score:.4f}")
                except Exception as e:
                    st.error(f"错误: {e}")
        
        st.subheader("Sent2Vec: 句子相似度计算")
        st.write("使用FastText词向量进行平均池化得到句子向量")
        
        sent1 = st.text_area("句子1:", value="Natural language processing is a subfield of artificial intelligence.", height=80, key="sent1")
        sent2 = st.text_area("句子2:", value="Machine learning algorithms process text data effectively.", height=80, key="sent2")
        
        if st.button("计算句子相似度", key="sent_sim_btn"):
            ft_model = st.session_state['ft_model']
            
            def get_sentence_vector(sentence, model):
                tokens = word_tokenize(sentence.lower())
                vectors = []
                for token in tokens:
                    try:
                        vectors.append(model.wv[token])
                    except:
                        continue
                if vectors:
                    return np.mean(vectors, axis=0)
                return np.zeros(model.vector_size)
            
            def cosine_similarity(v1, v2):
                return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            
            vec1 = get_sentence_vector(sent1, ft_model)
            vec2 = get_sentence_vector(sent2, ft_model)
            
            similarity = cosine_similarity(vec1, vec2)
            
            st.write(f"句子相似度: **{similarity:.4f}**")
            st.write(f"句子1向量维度: {vec1.shape}")
            st.write(f"句子2向量维度: {vec2.shape}")
