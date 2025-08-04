#!/usr/bin/env python3
"""
Direct Concept Pipeline Runner
简单的运行脚本
"""

import sys
import os
import argparse
from pathlib import Path

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')

def main():
    parser = argparse.ArgumentParser(description='Run Direct Concept Extraction Pipeline')
    
    # 必需参数
    parser.add_argument('--data_dir', required=True, help='Data directory path')
    parser.add_argument('--filename_pattern', required=True, help='Filename pattern to match')
    
    # 模型参数
    parser.add_argument('--model', default='gpt-4o', help='Model name (default: gpt-4o)')
    parser.add_argument('--api_key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    # 输出参数
    parser.add_argument('--output_dir', default='NewWork/output', help='Output directory')
    parser.add_argument('--output_name', help='Custom output name')
    
    # 处理参数
    parser.add_argument('--mode', choices=['passage_concept', 'hierarchical_concept'], 
                       default='passage_concept', help='Extraction mode')
    parser.add_argument('--language', choices=['en', 'zh'], default='en', help='Language')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Text chunk size')
    
    # 运行模式
    parser.add_argument('--extract_only', action='store_true', help='Only extract concepts')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # 设置API key
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    elif not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: Please provide --api_key or set OPENAI_API_KEY environment variable")
        return 1
    
    try:
        # 导入依赖
        from openai import OpenAI
        from atlas_rag.llm_generator import LLMGenerator
        from direct_concept_pipeline import DirectConceptPipeline, create_default_config
        
        print(f"🚀 Starting Direct Concept Pipeline")
        print(f"📁 Data directory: {args.data_dir}")
        print(f"🔍 Filename pattern: {args.filename_pattern}")
        print(f"🤖 Model: {args.model}")
        print(f"🌍 Language: {args.language}")
        print(f"⚙️ Mode: {args.mode}")
        
        # 设置模型
        client = OpenAI()
        model = LLMGenerator(client, model_name=args.model)
        
        # 创建配置
        config = create_default_config(
            model_path=args.model,
            data_directory=args.data_dir,
            filename_pattern=args.filename_pattern,
            output_directory=args.output_dir,
            extraction_mode=args.mode,
            language=args.language
        )
        
        # 更新配置
        config.batch_size_concept = args.batch_size
        config.text_chunk_size = args.chunk_size
        config.debug_mode = args.debug
        
        # 创建pipeline
        pipeline = DirectConceptPipeline(model, config)
        
        # 运行pipeline
        if args.extract_only:
            print("\n🔄 Running concept extraction only...")
            outputs = pipeline.run_extraction_only()
        else:
            print("\n🔄 Running full pipeline...")
            outputs = pipeline.run_full_pipeline(args.output_name)
        
        print("\n✅ Pipeline completed successfully!")
        print("\n📁 Generated files:")
        for key, path in outputs.items():
            print(f"   📄 {key}: {path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 