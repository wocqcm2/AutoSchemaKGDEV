#!/usr/bin/env python3
"""
使用配置文件运行NewWork Pipeline
这是使用config.json配置文件的简化版本
"""

import os
import sys
import argparse
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from config_loader import ConfigLoader, create_model_client, quick_setup
from direct_concept_pipeline import DirectConceptPipeline


def run_simple_example():
    """简单示例：使用配置文件的最小示例"""
    print("🔥 Simple Example with Config File")
    print("=" * 50)
    
    try:
        # 1. 加载配置和模型（一行代码）
        config_loader, model, model_name = quick_setup()
        
        print(f"✅ Loaded model: {model_name}")
        
        # 2. 创建pipeline配置
        pipeline_config = config_loader.get_direct_concept_config(
            filename_pattern="Dulce_test",  # 使用小测试文件
            output_name="simple_test"
        )
        
        # 3. 运行pipeline
        pipeline = DirectConceptPipeline(model, pipeline_config)
        outputs = pipeline.run_full_pipeline("dulce_simple")
        
        print("\n🎉 Success! Generated files:")
        for key, path in outputs.items():
            print(f"   📄 {key}: {path}")
            
        return True
        
    except Exception as e:
        print(f"❌ Simple example failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_with_different_models():
    """使用不同模型的示例"""
    print("\n🔥 Testing Different Models")
    print("=" * 50)
    
    config_loader = ConfigLoader()
    
    # 显示可用模型
    models = config_loader.list_available_models()
    print("Available models:")
    for short_name, full_name in models.items():
        print(f"  {short_name}: {full_name}")
    
    # 测试Qwen模型
    try:
        print(f"\n🤖 Testing Qwen model...")
        model = create_model_client(config_loader, "qwen_235b")
        
        pipeline_config = config_loader.get_direct_concept_config(
            model_name="qwen_235b",
            filename_pattern="Dulce_test",
            output_name="qwen_test",
            batch_size_concept=4,  # 大模型用小batch
            temperature=0.1
        )
        
        pipeline = DirectConceptPipeline(model, pipeline_config)
        outputs = pipeline.run_full_pipeline("dulce_qwen")
        
        print("✅ Qwen model test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Qwen model test failed: {e}")
        return False


def run_chinese_example():
    """中文数据处理示例"""
    print("\n🔥 Chinese Text Processing with Config")
    print("=" * 50)
    
    try:
        config_loader, model, model_name = quick_setup()
        
        # 中文配置
        pipeline_config = config_loader.get_direct_concept_config(
            filename_pattern="RomanceOfTheThreeKingdom-zh-CN",
            output_name="chinese_test",
            extraction_mode="hierarchical_concept",  # 覆盖默认设置
            language="zh",  # 覆盖默认设置
            batch_size_concept=6
        )
        
        pipeline = DirectConceptPipeline(model, pipeline_config)
        outputs = pipeline.run_full_pipeline("three_kingdoms")
        
        print("✅ Chinese processing successful!")
        for key, path in outputs.items():
            print(f"   📄 {key}: {path}")
            
        return True
        
    except Exception as e:
        print(f"❌ Chinese processing failed: {e}")
        return False


def run_large_document_example():
    """大文档处理示例"""
    print("\n🔥 Large Document Processing with Config")
    print("=" * 50)
    
    try:
        config_loader, model, model_name = quick_setup()
        
        # 大文档配置
        pipeline_config = config_loader.get_direct_concept_config(
            filename_pattern="Apple_Environmental_Progress_Report_2024",
            output_name="apple_report",
            batch_size_concept=3,    # 大文档用小batch
            text_chunk_size=600,     # 小chunk
            chunk_overlap=150,       # 大重叠
            min_concept_frequency=2, # 过滤低频概念
            max_workers=2           # 减少并发
        )
        
        print("⚠️ This will process a large document and may take some time...")
        choice = input("Continue? (y/n): ").lower().strip()
        
        if choice != 'y':
            print("Skipped large document processing")
            return True
        
        pipeline = DirectConceptPipeline(model, pipeline_config)
        outputs = pipeline.run_full_pipeline("apple_environmental")
        
        print("✅ Large document processing successful!")
        for key, path in outputs.items():
            print(f"   📄 {key}: {path}")
            
        return True
        
    except Exception as e:
        print(f"❌ Large document processing failed: {e}")
        return False


def interactive_mode():
    """交互模式"""
    print("\n🔥 Interactive Mode")
    print("=" * 50)
    
    config_loader = ConfigLoader()
    
    # 显示配置摘要
    config_loader.print_config_summary()
    
    while True:
        print("\n📋 Available Actions:")
        print("1. Run simple test")
        print("2. Try different models")  
        print("3. Process Chinese text")
        print("4. Process large document")
        print("5. Show config")
        print("6. Quit")
        
        choice = input("\nSelect action (1-6): ").strip()
        
        if choice == '1':
            run_simple_example()
        elif choice == '2':
            run_with_different_models()
        elif choice == '3':
            run_chinese_example()
        elif choice == '4':
            run_large_document_example()
        elif choice == '5':
            config_loader.print_config_summary()
        elif choice == '6':
            break
        else:
            print("❌ Invalid choice, please select 1-6")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run NewWork Pipeline with Config File')
    
    parser.add_argument('--config', default='config.json', help='Config file path')
    parser.add_argument('--model', help='Model name from config')
    parser.add_argument('--data', help='Data filename pattern')
    parser.add_argument('--output', help='Output name')
    parser.add_argument('--mode', choices=['simple', 'models', 'chinese', 'large', 'interactive'], 
                       default='simple', help='Run mode')
    parser.add_argument('--batch-size', type=int, help='Batch size override')
    parser.add_argument('--language', choices=['en', 'zh'], help='Language override')
    
    args = parser.parse_args()
    
    print("🌟 NewWork Pipeline with Configuration")
    print("=" * 60)
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        print("Please make sure config.json exists in the current directory")
        return 1
    
    try:
        # 根据模式运行
        if args.mode == 'simple':
            success = run_simple_example()
        elif args.mode == 'models':
            success = run_with_different_models()
        elif args.mode == 'chinese':
            success = run_chinese_example()
        elif args.mode == 'large':
            success = run_large_document_example()
        elif args.mode == 'interactive':
            interactive_mode()
            success = True
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return 1
        
        if success:
            print("\n🎊 Pipeline completed successfully!")
            return 0
        else:
            print("\n❌ Pipeline failed!")
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())