# region imports
from AlgorithmImports import *
# endregion
# 系统资源监控模块 - QuantConnect兼容版本
import gc
import sys
import time

# QuantConnect环境下不支持psutil，使用基础监控
PSUTIL_AVAILABLE = False

class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self, algorithm=None):
        self.algorithm = algorithm
        self.start_time = time.time()
        
        # QuantConnect兼容的基础监控
        self.initial_memory = 0
    
    def get_resource_info(self, phase=""):
        """获取当前资源使用信息 - QuantConnect兼容版本"""
        try:
            resource_info = {
                'phase': phase,
                'timestamp': time.time(),
                'uptime': time.time() - self.start_time
            }
            
            # QuantConnect环境下的基础监控
            resource_info.update({
                'objects_in_memory': len(gc.get_objects()),
                'python_version': sys.version.split()[0],
                'platform': sys.platform
            })
            
            # 垃圾回收统计
            try:
                gc_stats = gc.get_stats()
                if gc_stats and len(gc_stats) >= 3:
                    resource_info.update({
                        'gc_gen0_collections': gc_stats[0]['collections'],
                        'gc_gen1_collections': gc_stats[1]['collections'],
                        'gc_gen2_collections': gc_stats[2]['collections']
                    })
            except Exception:
                pass
            
            return resource_info
            
        except Exception as e:
            return {
                'phase': phase,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def log_resources(self, phase="", algorithm=None):
        """记录资源使用情况到算法日志"""
        algo = algorithm or self.algorithm
        if not algo:
            return
        
        try:
            info = self.get_resource_info(phase)
            
            if 'error' in info:
                algo.log_debug(f"[{phase}] Resource monitoring error: {info['error']}")
                return
            
            algo.log_debug(f"[{phase}]System Resources (Uptime: {info['uptime']:.1f}s):")
            
            # QuantConnect兼容的基础监控日志
            algo.log_debug(f"  Objects in memory: {info['objects_in_memory']}")
            algo.log_debug(f"  Python: {info['python_version']} on {info['platform']}")
            
            if 'gc_gen0_collections' in info:
                algo.log_debug(f"  GC Collections: Gen0={info['gc_gen0_collections']}, Gen1={info['gc_gen1_collections']}, Gen2={info['gc_gen2_collections']}")
            
        except Exception as e:
            algo.log_debug(f"[{phase}]Error logging resources: {e}")
    

    
    def force_garbage_collection(self):
        """强制垃圾回收并返回回收的对象数量"""
        try:
            objects_before = len(gc.get_objects())
            collected = gc.collect()
            objects_after = len(gc.get_objects())
            
            return {
                'objects_before': objects_before,
                'objects_after': objects_after,
                'objects_freed': objects_before - objects_after,
                'cycles_collected': collected
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def is_available():
        """检查资源监控是否可用"""
        return PSUTIL_AVAILABLE
    
    @staticmethod
    def get_availability_info():
        """获取可用性信息 - QuantConnect兼容版本"""
        return {
            'available': False,
            'fallback': True,
            'features': ['garbage_collection', 'basic_info'],
            'note': 'QuantConnect basic monitoring only'
        } 