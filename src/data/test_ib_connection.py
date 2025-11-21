"""
IB Gateway连接测试脚本
用于验证IB Gateway连接是否正常
"""

import logging
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data import IBConnector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_connection(host='127.0.0.1', port=4001, client_id=1):
    """
    测试IB Gateway连接
    
    Args:
        host: IB Gateway主机地址
        port: IB Gateway端口
        client_id: 客户端ID
    """
    print("\n" + "=" * 70)
    print("IB Gateway连接测试")
    print("=" * 70)
    
    print(f"\n连接参数:")
    print(f"  主机: {host}")
    print(f"  端口: {port}")
    print(f"  客户端ID: {client_id}")
    
    port_info = {
        4001: "TWS Live Trading (实盘)",
        4001: "TWS Paper Trading (模拟盘)",
        4003: "IB Gateway Live Trading (实盘)",
        4004: "IB Gateway Paper Trading (模拟盘)"
    }
    print(f"  端口说明: {port_info.get(port, '未知')}")
    
    print("\n" + "-" * 70)
    print("开始测试...")
    print("-" * 70)
    
    try:
        # 创建连接器
        print("\n[1/5] 创建IB连接器...")
        connector = IBConnector(host=host, port=port, client_id=client_id, timeout=10)
        print("✓ 连接器创建成功")
        
        # 尝试连接
        print("\n[2/5] 连接到IB Gateway...")
        success = connector.connect()
        
        if not success:
            print("✗ 连接失败！")
            print("\n可能的原因：")
            print("  1. IB Gateway或TWS未启动")
            print("  2. 端口号不正确")
            print("  3. API连接未启用")
            print("  4. 防火墙阻止连接")
            print("\n解决方法：")
            print("  1. 启动IB Gateway或TWS")
            print("  2. 在IB Gateway/TWS中：配置 → API → 设置")
            print("     - 勾选'启用ActiveX和Socket客户端'")
            print("     - 设置Socket端口（4001/4001/4003/4004）")
            print("     - 添加信任的IP地址（127.0.0.1）")
            return False
        
        print("✓ 成功连接到IB Gateway")
        
        # 获取账户信息
        print("\n[3/5] 获取账户信息...")
        accounts = connector.ib.managedAccounts()
        if accounts:
            print(f"✓ 可用账户: {accounts}")
            for account in accounts:
                print(f"  - {account}")
        else:
            print("⚠ 未找到账户")
        
        # 获取服务器时间
        print("\n[4/5] 获取服务器时间...")
        server_time = connector.get_current_time()
        if server_time:
            print(f"✓ IB服务器时间: {server_time}")
            
            # 计算时差
            from datetime import datetime
            local_time = datetime.now()
            time_diff = (local_time - server_time.replace(tzinfo=None)).total_seconds()
            print(f"  本地时间: {local_time}")
            print(f"  时差: {time_diff:.1f} 秒")
        else:
            print("⚠ 无法获取服务器时间")
        
        # 测试合约验证
        print("\n[5/5] 测试合约验证...")
        from src.data import create_futures_contract
        
        # 测试ES合约
        print("  测试ES期货合约...")
        es_contract = create_futures_contract('ES', 'CME')
        qualified = connector.qualify_contract(es_contract)
        
        if qualified:
            print(f"  ✓ ES合约验证成功: {qualified.symbol} {qualified.secType} {qualified.exchange}")
            print(f"    合约ID: {qualified.conId}")
            print(f"    本地代码: {qualified.localSymbol}")
        else:
            print("  ⚠ ES合约验证失败（可能需要市场数据订阅）")
        
        # 断开连接
        print("\n断开连接...")
        connector.disconnect()
        print("✓ 已断开连接")
        
        # 测试结果
        print("\n" + "=" * 70)
        print("测试结果: ✓ 成功")
        print("=" * 70)
        print("\nIB Gateway连接正常！可以开始使用数据下载功能。")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        
        import traceback
        print("\n详细错误信息:")
        print(traceback.format_exc())
        
        print("\n" + "=" * 70)
        print("测试结果: ✗ 失败")
        print("=" * 70)
        
        print("\n故障排查步骤:")
        print("1. 检查IB Gateway/TWS是否运行")
        print("2. 检查API设置:")
        print("   - 配置 → API → 设置")
        print("   - 启用'ActiveX和Socket客户端'")
        print("   - 检查Socket端口号")
        print("   - 添加127.0.0.1到信任IP")
        print("3. 检查防火墙设置")
        print("4. 尝试重启IB Gateway/TWS")
        
        return False


def interactive_test():
    """交互式测试"""
    print("\n" + "=" * 70)
    print("IB Gateway连接测试 - 交互模式")
    print("=" * 70)
    
    print("\n请选择连接类型:")
    print("1. TWS Paper Trading (端口 4001) - 推荐用于测试")
    print("2. TWS Live Trading (端口 4001)")
    print("3. IB Gateway Paper Trading (端口 4004)")
    print("4. IB Gateway Live Trading (端口 4003)")
    print("5. 自定义端口")
    
    choice = input("\n请输入选项 (1-5，默认1): ").strip() or "1"
    
    port_map = {
        "1": 4001,
        "2": 4001,
        "3": 4004,
        "4": 4003
    }
    
    if choice in port_map:
        port = port_map[choice]
    elif choice == "5":
        port = int(input("请输入端口号: "))
    else:
        print("无效选项，使用默认端口4001")
        port = 4001
    
    host = input("请输入主机地址 (默认 127.0.0.1): ").strip() or "127.0.0.1"
    client_id = input("请输入客户端ID (默认 1): ").strip() or "1"
    client_id = int(client_id)
    
    print("\n准备连接...")
    input("按Enter键开始测试...")
    
    return test_connection(host=host, port=port, client_id=client_id)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='IB Gateway连接测试')
    parser.add_argument('--host', default='127.0.0.1', help='IB Gateway主机地址')
    parser.add_argument('--port', type=int, default=4001, help='IB Gateway端口')
    parser.add_argument('--client-id', type=int, default=1, help='客户端ID')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互模式')
    
    args = parser.parse_args()
    
    if args.interactive:
        # 交互模式
        success = interactive_test()
    else:
        # 命令行模式
        success = test_connection(
            host=args.host,
            port=args.port,
            client_id=args.client_id
        )
    
    # 返回退出码
    sys.exit(0 if success else 1)