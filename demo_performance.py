#!/usr/bin/env python3
"""
Performance demonstration script for concurrent AI enhancement
Shows the difference between serial and concurrent processing modes
"""
import os
import sys
import json
import time
import tempfile
from unittest.mock import patch, MagicMock

# Add the ai directory to the path
sys.path.insert(0, '/home/runner/work/daae/daae/ai')

def create_demo_papers(count=10):
    """Create demo papers for performance testing"""
    papers = []
    for i in range(count):
        paper = {
            "id": f"demo.{i+1:03d}",
            "title": f"Demo Paper {i+1}: Advanced Research in AI and Machine Learning",
            "summary": f"This is demo paper number {i+1} about advanced artificial intelligence research. "
                      f"It covers topics including machine learning, deep learning, neural networks, "
                      f"natural language processing, computer vision, and automated reasoning. "
                      f"The paper presents novel algorithms and methodologies for solving complex problems "
                      f"in the field of artificial intelligence and demonstrates significant improvements "
                      f"over existing state-of-the-art approaches.",
            "authors": [f"Demo Author {i+1}", f"Co-Author {i+1}"],
            "categories": ["cs.AI", "cs.LG"]
        }
        papers.append(paper)
    return papers

def mock_ai_response(delay=0.5):
    """Create a mock AI response with simulated processing delay"""
    time.sleep(delay)  # Simulate AI processing time
    
    class MockResponse:
        def model_dump(self):
            return {
                'title_translation': f'演示论文标题翻译 {time.time():.0f}',
                'tldr': 'Too long; didn\'t read 演示摘要',
                'motivation': '本研究的动机是推进人工智能领域的发展',
                'method': '我们采用了先进的机器学习方法和深度神经网络',
                'result': '实验结果显示我们的方法优于现有技术',
                'conclusion': '本研究为AI领域做出了重要贡献',
                'translation': '这是一篇关于人工智能和机器学习高级研究的演示论文',
                'summary': '本文提出了创新的AI算法和方法论',
                'keywords': '人工智能, 机器学习, 深度学习, 神经网络',
                'comments': '这是一篇高质量的研究论文，具有重要的学术价值'
            }
    return MockResponse()

def demo_serial_processing(papers, processing_delay=0.5):
    """Demonstrate serial processing performance"""
    print(f"\n🔄 串行处理模式演示 (处理 {len(papers)} 篇论文)")
    print("=" * 50)
    
    start_time = time.time()
    results = []
    
    for i, paper in enumerate(papers):
        print(f"📄 处理论文 {i+1}/{len(papers)}: {paper['id']}")
        
        # Simulate AI processing with delay
        response = mock_ai_response(processing_delay)
        result = paper.copy()
        result['AI'] = response.model_dump()
        results.append(result)
        
        # Show progress
        elapsed = time.time() - start_time
        rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
        print(f"   ⏱️  已用时: {elapsed:.1f}秒, 处理速度: {rate:.1f} 篇/分钟")
    
    total_time = time.time() - start_time
    final_rate = len(papers) / (total_time / 60) if total_time > 0 else 0
    
    print(f"\n✅ 串行处理完成!")
    print(f"   📊 总耗时: {total_time:.1f} 秒")
    print(f"   🚀 平均速度: {final_rate:.1f} 篇/分钟")
    print(f"   📈 成功处理: {len(results)} 篇")
    
    return results, total_time

def demo_concurrent_processing(papers, num_workers=3, processing_delay=0.5):
    """Demonstrate concurrent processing performance"""
    print(f"\n⚡ 并发处理模式演示 (处理 {len(papers)} 篇论文，{num_workers} 个worker)")
    print("=" * 50)
    
    # Set up mock environment
    os.environ['GOOGLE_API_KEYS'] = ','.join([f'demo_key_{i+1}' for i in range(num_workers)])
    os.environ['MODEL_PRIORITY_LIST'] = 'gemini-1.5-flash'
    os.environ['LANGUAGE'] = 'Chinese'
    
    try:
        from enhance_concurrent import PaperProcessor, WorkerConfig
        from langchain.prompts import ChatPromptTemplate
        
        # Create worker configs
        worker_configs = []
        for i in range(num_workers):
            config = WorkerConfig(
                worker_id=i,
                key_name=f"演示密钥_{i+1}",
                api_key=f"demo_key_{i+1}",
                model_name="gemini-1.5-flash",
                rpm_limit=120  # High RPM for demo
            )
            worker_configs.append(config)
        
        # Mock template and chain
        mock_template = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = lambda x: mock_ai_response(processing_delay)
        mock_template.__or__ = MagicMock(return_value=mock_chain)
        
        # Create processor
        processor = PaperProcessor(
            worker_configs=worker_configs,
            prompt_template=mock_template,
            retries=1,
            timeout=0,
            language="Chinese"
        )
        
        # Mock the worker chains
        for i in range(num_workers):
            processor.worker_chains[i] = mock_chain
        
        # Process papers
        start_time = time.time()
        results = processor.process_papers_concurrent(papers, max_workers=num_workers)
        total_time = time.time() - start_time
        
        final_rate = len(results) / (total_time / 60) if total_time > 0 else 0
        
        print(f"\n✅ 并发处理完成!")
        print(f"   📊 总耗时: {total_time:.1f} 秒")
        print(f"   🚀 平均速度: {final_rate:.1f} 篇/分钟")
        print(f"   📈 成功处理: {processor.stats.successful_papers} 篇")
        print(f"   ❌ 处理失败: {processor.stats.failed_papers} 篇")
        
        return results, total_time
        
    except Exception as e:
        print(f"❌ 并发处理演示失败: {e}")
        return [], 0

def performance_comparison():
    """Run performance comparison between serial and concurrent modes"""
    print("🔬 AI增强脚本性能对比演示")
    print("=" * 60)
    
    # Create demo data
    num_papers = 8
    num_workers = 3
    processing_delay = 0.3  # Simulate 0.3s AI processing per paper
    
    print(f"📋 测试参数:")
    print(f"   📄 论文数量: {num_papers}")
    print(f"   ⚙️  Worker数量: {num_workers}")
    print(f"   ⏳ 模拟AI处理延迟: {processing_delay}s/篇")
    
    papers = create_demo_papers(num_papers)
    
    # Test serial processing
    serial_results, serial_time = demo_serial_processing(papers, processing_delay)
    
    # Test concurrent processing
    concurrent_results, concurrent_time = demo_concurrent_processing(papers, num_workers, processing_delay)
    
    # Performance comparison
    if serial_time > 0 and concurrent_time > 0:
        speedup = serial_time / concurrent_time
        serial_rate = num_papers / (serial_time / 60)
        concurrent_rate = num_papers / (concurrent_time / 60)
        
        print(f"\n📊 性能对比结果")
        print("=" * 40)
        print(f"📈 串行模式速度: {serial_rate:.1f} 篇/分钟")
        print(f"⚡ 并发模式速度: {concurrent_rate:.1f} 篇/分钟")
        print(f"🚀 性能提升倍数: {speedup:.1f}x")
        print(f"⏱️  时间节省: {((serial_time - concurrent_time) / serial_time * 100):.1f}%")
        
        if speedup > 1.5:
            print("✅ 并发处理显著提升了处理效率！")
        else:
            print("⚠️  性能提升有限，可能需要调整参数")
    
    print(f"\n💡 实际应用建议:")
    print(f"   • 对于 {num_workers} 个API密钥，理论最大提升可达 {num_workers}x")
    print(f"   • 实际提升受网络延迟、API响应时间等因素影响")
    print(f"   • 推荐在有多个API密钥时启用并发模式")
    print(f"   • 可根据实际情况调整 --max-workers 和 --rpm-limit 参数")

if __name__ == "__main__":
    performance_comparison()