#!/usr/bin/env python3
"""
Direct Concept Pipeline Usage Example
直接概念提取pipeline的使用示例
"""

import sys
import os
from openai import OpenAI
from transformers import pipeline
from configparser import ConfigParser

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from direct_concept_pipeline import DirectConceptPipeline, create_default_config


def setup_openai_model():
    """设置OpenAI模型"""
    # 读取配置文件
    config = ConfigParser()
    config.read('config.ini')
    
    # 创建OpenAI客户端
    client = OpenAI(
        api_key=config['settings']['OPENAI_API_KEY']
    )
    
    # 导入LLMGenerator
    sys.path.append('..')  # 添加上级目录以导入atlas_rag
    from atlas_rag.llm_generator import LLMGenerator
    
    model_name = "gpt-4o"
    llm_generator = LLMGenerator(client, model_name=model_name)
    
    return llm_generator, model_name


def setup_local_model():
    """设置本地Transformers模型"""
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # 创建本地模型pipeline
    client = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
    )
    
    # 导入LLMGenerator
    sys.path.append('..')
    from atlas_rag.llm_generator import LLMGenerator
    
    llm_generator = LLMGenerator(client, model_name=model_name)
    
    return llm_generator, model_name


def example_basic_usage():
    """基本使用示例"""
    print("🔥 Basic Usage Example")
    print("=" * 50)
    
    # 1. 设置模型（选择一种）
    try:
        # 选项1: 使用OpenAI API
        model, model_name = setup_openai_model()
        print(f"✅ Using OpenAI model: {model_name}")
    except:
        try:
            # 选项2: 使用本地模型
            model, model_name = setup_local_model()
            print(f"✅ Using local model: {model_name}")
        except Exception as e:
            print(f"❌ Failed to setup model: {e}")
            print("Please check your model configuration.")
            return
    
    # 2. 创建配置
    config = create_default_config(
        model_path=model_name,
        data_directory="../example_data",  # 使用项目中的示例数据
        filename_pattern="Dulce",  # 处理Dulce数据
        output_directory="NewWork/output/basic_example",
        extraction_mode="passage_concept",  # 或者 "hierarchical_concept"
        language="en"  # 或者 "zh"
    )
    
    # 3. 创建pipeline
    pipeline_instance = DirectConceptPipeline(model, config)
    
    # 4. 运行完整pipeline
    try:
        outputs = pipeline_instance.run_full_pipeline("dulce_concepts")
        
        print("\n🎉 Success! Generated files:")
        for key, path in outputs.items():
            print(f"   📄 {key}: {path}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")


def example_step_by_step():
    """分步执行示例"""
    print("\n🔥 Step-by-Step Execution Example")
    print("=" * 50)
    
    # 设置模型
    try:
        model, model_name = setup_openai_model()
    except:
        model, model_name = setup_local_model()
    
    # 创建配置
    config = create_default_config(
        model_path=model_name,
        data_directory="../example_data",
        filename_pattern="Apple_Environmental",  # 处理苹果环境报告
        output_directory="NewWork/output/step_example",
        extraction_mode="hierarchical_concept",  # 使用层次概念模式
        language="en"
    )
    
    pipeline_instance = DirectConceptPipeline(model, config)
    
    try:
        # 步骤1: 仅提取概念
        print("\n🚀 Step 1: Extract concepts only...")
        extraction_results = pipeline_instance.run_extraction_only()
        
        print(f"✅ Concepts extracted to: {extraction_results['concepts_csv']}")
        print(f"✅ Relationships extracted to: {extraction_results['relationships_csv']}")
        
        # 步骤2: 从CSV构建图
        print("\n🔧 Step 2: Build graph from CSV...")
        graph_results = pipeline_instance.run_graph_only(
            extraction_results['concepts_csv'],
            extraction_results['relationships_csv'],
            "apple_environmental_graph"
        )
        
        print(f"✅ Graph saved to: {graph_results['graphml_file']}")
        print(f"✅ Statistics: {graph_results['statistics']}")
        
    except Exception as e:
        print(f"❌ Step-by-step execution failed: {e}")


def example_chinese_text():
    """中文文本处理示例"""
    print("\n🔥 Chinese Text Processing Example")
    print("=" * 50)
    
    # 创建中文示例数据
    os.makedirs("NewWork/example_data", exist_ok=True)
    
    chinese_example = {
        "text": """
        人工智能是计算机科学的一个分支，它致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。
        人工智能的研究领域包括机器学习、深度学习、自然语言处理、计算机视觉等。
        机器学习是人工智能的核心技术之一，通过算法使计算机能够自动学习和改进。
        深度学习是机器学习的一个子领域，使用神经网络来模拟人脑的工作方式。
        """,
        "metadata": {
            "file_id": "ai_intro_zh",
            "lang": "zh",
            "title": "人工智能简介"
        }
    }
    
    # 保存中文示例数据
    import json
    with open("NewWork/example_data/chinese_example.json", "w", encoding="utf-8") as f:
        json.dump([chinese_example], f, ensure_ascii=False, indent=2)
    
    # 设置模型
    try:
        model, model_name = setup_openai_model()
    except:
        model, model_name = setup_local_model()
    
    # 创建中文配置
    config = create_default_config(
        model_path=model_name,
        data_directory="NewWork/example_data",
        filename_pattern="chinese_example",
        output_directory="NewWork/output/chinese_example",
        extraction_mode="passage_concept",
        language="zh"  # 使用中文
    )
    
    pipeline_instance = DirectConceptPipeline(model, config)
    
    try:
        outputs = pipeline_instance.run_full_pipeline("chinese_ai_concepts")
        
        print("\n🎉 Chinese text processing completed!")
        for key, path in outputs.items():
            print(f"   📄 {key}: {path}")
        
    except Exception as e:
        print(f"❌ Chinese text processing failed: {e}")


def example_custom_config():
    """自定义配置示例"""
    print("\n🔥 Custom Configuration Example")
    print("=" * 50)
    
    # 设置模型
    try:
        model, model_name = setup_openai_model()
    except:
        model, model_name = setup_local_model()
    
    # 导入配置类
    from direct_concept_config import DirectConceptConfig
    
    # 创建自定义配置
    custom_config = DirectConceptConfig(
        # 模型配置
        model_path=model_name,
        max_new_tokens=3072,  # 更大的输出长度
        temperature=0.3,      # 更低的温度，更确定性
        max_workers=5,        # 更多并行workers
        
        # 数据配置
        data_directory="../example_data",
        filename_pattern="CICGPC",
        output_directory="NewWork/output/custom_example",
        
        # 处理配置
        batch_size_concept=4,    # 较小的batch size
        text_chunk_size=800,     # 较小的文本块
        chunk_overlap=150,       # 更大的重叠
        
        # 概念提取配置
        extraction_mode="hierarchical_concept",  # 层次概念模式
        language="en",
        
        # 图构建配置
        include_abstraction_levels=True,
        include_hierarchical_relations=True,
        min_concept_frequency=2,  # 只保留出现2次以上的概念
        
        # 质量控制
        normalize_concept_names=True,
        filter_low_quality_concepts=True,
        
        # 调试配置
        debug_mode=True,
        record_usage=True,
        save_intermediate_results=True
    )
    
    pipeline_instance = DirectConceptPipeline(model, custom_config)
    
    try:
        outputs = pipeline_instance.run_full_pipeline("cicgpc_concepts")
        
        print("\n🎉 Custom configuration processing completed!")
        for key, path in outputs.items():
            print(f"   📄 {key}: {path}")
        
    except Exception as e:
        print(f"❌ Custom configuration processing failed: {e}")


def main():
    """主函数 - 运行所有示例"""
    print("🌟 Direct Concept Pipeline Examples")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs("NewWork/output", exist_ok=True)
    
    try:
        # 示例1: 基本使用
        example_basic_usage()
        
        # 示例2: 分步执行
        example_step_by_step()
        
        # 示例3: 中文文本处理
        example_chinese_text()
        
        # 示例4: 自定义配置
        example_custom_config()
        
        print("\n🎊 All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 