import importlib.util
import sys

# 项目所需的所有核心依赖（含版本要求）
required_packages = {
    'tensorflow': '2.20.0',
    'opencv-python': '4.8.0.76',
    'mtcnn': '0.1.0',
    'dlib': '19.24.6',
    'pandas': '2.3.3',
    'numpy': '2.3.3',
    'matplotlib': '3.10.6'
}

# 包名与导入名的映射（有些包安装名和导入名不一样）
import_name_map = {
    'opencv-python': 'cv2',
    'tensorflow': 'tensorflow',
    'mtcnn': 'mtcnn',
    'dlib': 'dlib',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib'
}

print('='*60)
print('📦 项目核心依赖安装状态检查结果')
print('='*60)

for pkg_name, required_ver in required_packages.items():
    import_name = import_name_map[pkg_name]
    try:
        # 检查是否已安装
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f'❌ {pkg_name:<20} 未安装（要求版本：{required_ver}）')
        else:
            # 获取已安装版本
            module = importlib.import_module(import_name)
            installed_ver = getattr(module, '__version__', '未知版本')
            # 简单版本匹配
            if installed_ver.startswith(required_ver.split('.')[0] + '.' + required_ver.split('.')[1]):
                print(f'✅ {pkg_name:<20} 已安装（当前版本：{installed_ver}，要求版本：{required_ver}）')
            else:
                print(f'⚠️  {pkg_name:<20} 版本不匹配（当前版本：{installed_ver}，要求版本：{required_ver}）')
    except Exception as e:
        print(f'❌ {pkg_name:<20} 检查失败（可能存在安装损坏，建议重新安装）')

print('='*60)
print('💡 说明：标"✅"的无需处理，标"❌"或"⚠️"的后续需补充安装')