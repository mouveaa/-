# app.py 文件

# 标准库导入
import os
import sys

# 添加当前目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 第三方库导入
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 自定义模块导入
from ai_module import answer_question, generate_exercise
from knowledge_graph import KnowledgeGraph
from speech_recognition import load_model, speech_to_text
from data_analysis import LearningAnalytics

# 导入DeepSeek API模块（如果存在）
try:
    from deepseek_api import ask_deepseek, generate_exercises
except ImportError:
    # 定义兼容函数，在未找到模块时使用
    def ask_deepseek(question, api_key):
        return f"DeepSeek API模块未安装。问题: {question}"
    
    def generate_exercises(topic, difficulty, count, api_key):
        return f"DeepSeek API模块未安装。无法生成关于{topic}的{difficulty}练习题。"

# 初始化会话状态（所有初始化放在一起）
if 'api_key' not in st.session_state:
    # 尝试从环境变量获取API密钥
    st.session_state.api_key = os.environ.get("DEEPSEEK_API_KEY", "")

if 'student_id' not in st.session_state:
    st.session_state.student_id = 'student001'

if 'learning_data' not in st.session_state:
    st.session_state.learning_data = LearningAnalytics()
    # 添加一些样本数据
    st.session_state.learning_data.add_record('student001', '代数', 45, 7, 10)
    st.session_state.learning_data.add_record('student001', '几何', 60, 5, 10)
    st.session_state.learning_data.add_record('student001', '数学', 30, 8, 10)
    st.session_state.learning_data.add_record('student001', '英语', 50, 6, 10)
    st.session_state.learning_data.add_record('student001', '语文', 40, 9, 10)

if 'show_answer_area' not in st.session_state:
    st.session_state.show_answer_area = False

if 'current_exercises' not in st.session_state:
    st.session_state.current_exercises = ""

# 创建知识图谱初始化函数
def initialize_knowledge_graph():
    # 这里我们假设已经有Neo4j实例运行在localhost:7687
    # 如果你没有Neo4j，可以注释掉实际连接部分，仅使用模拟数据
    try:
        kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
        
        # 添加示例概念
        concepts = [
            ("数学", "学科"),
            ("代数", "数学"),
            ("几何", "数学"),
            ("方程", "代数"),
            ("一元二次方程", "方程")
        ]
        
        for concept, subject in concepts:
            kg.add_concept(concept, subject)
        
        # 添加关系
        relationships = [
            ("数学", "代数", "包含"),
            ("数学", "几何", "包含"),
            ("代数", "方程", "包含"),
            ("方程", "一元二次方程", "包含")
        ]
        
        for concept1, concept2, rel in relationships:
            kg.add_relationship(concept1, concept2, rel)
            
        return kg
    except Exception as e:
        st.error(f"知识图谱初始化失败: {e}")
        # 返回None或模拟对象
        return None

# 创建示例学习数据函数
def create_sample_learning_data():
    analytics = LearningAnalytics()
    
    # 添加一些示例记录
    sample_data = [
        ('student001', '数学', 30, 8, 10),
        ('student001', '代数', 45, 7, 10),
        ('student001', '几何', 60, 5, 10),
        ('student001', '语文', 40, 9, 10),
        ('student001', '英语', 50, 6, 10)
    ]
    
    for student_id, topic, time_spent, correct, total in sample_data:
        analytics.add_record(student_id, topic, time_spent, correct, total)
    
    return analytics

# 初始化示例数据
if 'initialized' not in st.session_state:
    st.session_state.learning_data = create_sample_learning_data()
    st.session_state.initialized = True

# 主应用程序
def main():
    st.title("智慧教育助手")
    st.sidebar.title("功能导航")
    
    menu = st.sidebar.radio("选择功能", ["首页", "学习分析", "智能问答", "练习生成", "语音助手"])
    
    if menu == "首页":
        st.write("欢迎使用智慧教育助手！")
        st.write("这是一个AI驱动的个性化学习平台，帮助您更高效地学习。")
        st.write("请从左侧菜单选择功能开始使用。")
        
        # 显示一些基本信息
        st.subheader("当前学生: Student001")
        st.subheader("已学习科目:")
        subjects = st.session_state.learning_data.data['topic'].unique()
        for subject in subjects:
            st.write(f"- {subject}")
    
    elif menu == "学习分析":
        st.subheader("学习分析")
    
        # 获取性能报告
        performance = st.session_state.learning_data.get_performance_report('student001')
    
        # 修改列名为中文
        performance_display = performance.copy()
        performance_display.columns = ['正确题数', '总题数', '耗时(分钟)', '准确率']
        
        # 显示数据表格
        st.write("学习表现数据:")
        st.dataframe(performance_display)
        
        import matplotlib.font_manager as fm
    
        # 尝试多种可能的中文字体，按优先级排列
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'AR PL UMing CN', 'WenQuanYi Micro Hei', 'Arial Unicode MS']

        # 查找系统中第一个可用的中文字体
        chinese_font = None
        for font in chinese_fonts:
            if any(f.name == font for f in fm.fontManager.ttflist):
                chinese_font = font
                break

        if chinese_font:
            plt.rcParams['font.sans-serif'] = [chinese_font] + plt.rcParams['font.sans-serif']
        else:
            st.warning("未找到支持中文的字体，图表中的中文可能显示为方块")

        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 创建可视化图表
        fig = plt.figure(figsize=(10, 6))
        plt.bar(performance.index, performance['accuracy'])
        plt.ylabel('准确率 (%)')
        plt.xlabel('学科')
        plt.title('学习表现分析')
        st.pyplot(fig)
        
        # 提供学习建议
        st.subheader("学习建议:")
        weakest_subject = performance['accuracy'].idxmin()
        st.write(f"- 建议加强 {weakest_subject} 的学习，当前准确率最低。")
        strongest_subject = performance['accuracy'].idxmax()
        st.write(f"- 在 {strongest_subject} 方面表现良好，可以尝试更高难度的内容。")
    
    elif menu == "智能问答":
        st.subheader("智能问答")
        
        # API密钥输入
        api_key = st.text_input("DeepSeek API密钥（首次使用后会保存）:", value=st.session_state.api_key, type="password")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        # 创建问题输入框
        user_question = st.text_input("请输入您的问题:")
        
        if user_question:
            try:
                # 对于特定问题的处理 - 保留原有的特定问题处理逻辑
                if "几何" in user_question and ("比较" in user_question or "差" in user_question):
                    # 从学习数据中获取几何的表现
                    performance = st.session_state.learning_data.get_performance_report('student001')
                    if '几何' in performance.index:
                        geo_accuracy = performance.loc['几何', 'accuracy']
                        avg_accuracy = performance['accuracy'].mean()
                        
                        if geo_accuracy < avg_accuracy:
                            answer = f"根据学习数据，几何科目的准确率为{geo_accuracy:.1f}%，低于平均准确率{avg_accuracy:.1f}%。建议加强几何学习。"
                        else:
                            answer = f"几何科目的准确率为{geo_accuracy:.1f}%，表现良好，与其他科目相比没有明显差距。"
                    else:
                        answer = "没有找到几何科目的学习数据。"
                else:
                    # 使用DeepSeek API处理一般问题
                    if st.session_state.api_key:
                        with st.spinner("思考中..."):
                            # 调用DeepSeek API获取答案
                            answer = ask_deepseek(user_question, st.session_state.api_key)
                    else:
                        # 如果没有API密钥，使用本地上下文
                        context = """
                        一元二次方程是代数中的基本方程类型，一般形式为ax²+bx+c=0，其中a≠0。
                        解一元二次方程可以使用公式法：x = (-b ± √(b² - 4ac)) / 2a。
                        判别式Δ = b² - 4ac决定了方程解的性质：
                        若Δ > 0，方程有两个不相等的实数解；
                        若Δ = 0，方程有两个相等的实数解（即有一个重根）；
                        若Δ < 0，方程没有实数解，但有两个共轭复数解。
                        """
                        # 调用本地AI模块获取答案
                        answer = answer_question(context, user_question)
                        st.info("目前使用的是本地问答系统。要获取更准确的回答，请输入DeepSeek API密钥。")
                
                st.write("回答:")
                st.markdown(answer)  # 使用markdown格式显示，支持更丰富的格式
                
            except Exception as e:
                st.error(f"处理问题时出错: {str(e)}")
                st.write("回答: 抱歉，目前无法回答您的问题，请稍后再试。")
    
    elif menu == "练习题生成":
        st.subheader("练习题生成")
        
        # API密钥输入（如果尚未输入）
        if not st.session_state.api_key:
            api_key = st.text_input("DeepSeek API密钥:", type="password")
            if api_key:
                st.session_state.api_key = api_key
        
        # 创建选项
        topic = st.selectbox("选择主题:", ["数学", "代数", "几何", "统计", "函数", "概率", "方程", "语文", "英语"])
        difficulty = st.select_slider("选择难度:", options=["简单", "中等", "困难"])
        count = st.slider("题目数量:", min_value=1, max_value=10, value=3)
        
        if st.button("生成练习题"):
            with st.spinner("正在生成练习题..."):
                try:
                    # 调用DeepSeek API生成练习题
                    exercises = generate_exercises(topic, difficulty, count, st.session_state.api_key)
                    
                    # 保存当前生成的练习题到会话状态
                    st.session_state.current_exercises = exercises
                    
                    st.markdown("### 生成的练习题:")
                    st.markdown(exercises)
                    
                    # 重置答题区显示状态
                    st.session_state.show_answer_area = False
                except Exception as e:
                    st.error(f"生成练习题时出错: {str(e)}")
                    st.markdown("无法生成练习题。请检查API密钥是否正确，或尝试以下解决方法：")
                    st.markdown("1. 确保已安装requests库: `pip install requests`")
                    st.markdown("2. 检查网络连接")
                    st.markdown("3. 验证API密钥是否有效")
        
        # 显示答题区按钮（仅当已生成练习题时显示）
        if 'current_exercises' in st.session_state and st.session_state.current_exercises:
            if not st.session_state.get("show_answer_area", False):
                if st.button("显示答题区"):
                    st.session_state.show_answer_area = True
            
            # 如果已点击"显示答题区"按钮，显示答题区
            if st.session_state.get("show_answer_area", False):
                # 添加答题区
                user_answer = st.text_area("请在这里写下您的答案:", height=200)
                
                if st.button("提交答案"):
                    st.write("答案已提交，系统记录中...")
                    
                    # 如果有API密钥，可以使用DeepSeek评估答案
                    if st.session_state.api_key:
                        with st.spinner("正在评估您的答案..."):
                            try:
                                evaluation_prompt = f"""
                                请评估以下答案的正确性和完整性。

                                题目：
                                {st.session_state.current_exercises}

                                学生答案：
                                {user_answer}

                                请提供：
                                1. 评分（满分100分）
                                2. 详细评语
                                3. 改进建议
                                """
                                evaluation = ask_deepseek(evaluation_prompt, st.session_state.api_key)
                                st.markdown("### 答案评估:")
                                st.markdown(evaluation)
                            except Exception as e:
                                st.error(f"评估答案时出错: {str(e)}")
                    else:
                        st.info("答案已记录。获取详细评估需要DeepSeek API密钥。")
    
    elif menu == "语音助手":
        st.subheader("语音助手")
        
        st.write("请上传一段WAV格式的语音文件:")
        uploaded_file = st.file_uploader("上传语音文件", type=["wav"])
        
        if uploaded_file is not None:
            # 保存上传的文件
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("开始识别"):
                try:
                    # 加载模型和处理语音
                    # 注意：根据你使用的具体语音识别方案，这里的代码可能需要调整
                    model = load_model()
                    if model:
                        text = speech_to_text("temp_audio.wav", model)
                        st.write("识别结果:")
                        st.write(text)
                        
                        # 作为问题进行处理
                        st.write("正在回答您的问题...")
                        context = "智能教育助手可以回答关于各个学科的基础知识问题。"
                        answer = answer_question(context, text)
                        st.write("回答:")
                        st.write(answer)
                    else:
                        st.error("模型加载失败")
                except Exception as e:
                    st.error(f"处理过程中出错: {e}")

# 运行主程序
if __name__ == "__main__":
    main()