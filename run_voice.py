# -*- coding: utf-8 -*-
"""
AI Voice Layer Standalone Entry Point
This is the dedicated entry point for ai_voice_layer.py only

Usage:
    python run_voice.py          # Mixed mode (text + voice input)
    python run_voice.py --text   # Text input only
    python run_voice.py --voice   # Voice input only
    python run_voice.py --test   # Quick test mode
    python run_voice.py --help   # Show this help

Interactive Mode Controls:
    • Text mode: Type command and press ENTER
    • Voice mode: Type 'voice' then ENTER, then speak your command
    • Type 'quit' to exit

Command Examples:
    • Add 1k resistor R1
    • Add 5V power source V1
    • Connect R1 and V1
    • Delete R1
"""

import sys
import os
import logging
import threading
import time

# Get project root (D:\python)
current_dir = os.path.dirname(os.path.abspath(__file__))
scr_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(scr_dir)

# Add project root to path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def capture_and_recognize_voice(ai_voice):
    """采集并识别语音，返回识别的文本"""
    try:
        if not ai_voice._microphone_initialized or ai_voice.microphone is None:
            return None
        
        print("\n[🎤] 正在监听... (请说话)")
        
        # 采集语音
        audio_data = ai_voice._capture_voice()
        if not audio_data:
            print("[❌] 未检测到语音")
            return None
        
        # 语音转文本
        text = ai_voice._speech_to_text(audio_data)
        if text:
            print(f"[✓] 识别到: {text}")
            return text
        else:
            print("[❌] 无法识别语音内容")
            return None
            
    except Exception as e:
        logger.error(f"语音识别错误: {e}")
        print(f"[❌] 语音识别错误: {e}")
        return None


def main(mode='mixed'):
    """
    Main function - Interactive mode with text and/or voice input
    
    Args:
        mode: 'mixed' (both), 'text' (text only), 'voice' (voice only)
    """
    print("=" * 60)
    print("AI Voice Layer - Interactive Mode")
    print("=" * 60)
    
    try:
        from scr.data import get_data_layer
        from scr.ai_voice import AIVoiceLayer
        
        print("\n[1] Module import success")
        
        # Initialize
        data_layer = get_data_layer()
        ai_voice = AIVoiceLayer(data_layer)
        
        print("[2] AI Voice Layer initialized")
        
        # Check microphone status
        mic_available = ai_voice._microphone_initialized
        if mic_available:
            print("[*] Microphone: Available ✓")
        else:
            print("[*] Microphone: Not available (text input only)")
            if mode == 'voice':
                print("[❌] Voice mode requires microphone!")
                return 1
        
        # ERNIE status
        if ai_voice.ernie_model and ai_voice.ernie_model.is_loaded:
            print("[*] ERNIE Model: Loaded ✓")
        else:
            print("[*] ERNIE Model: Not loaded (using rule-based parsing)")
        
        # Show usage instructions
        print("\n" + "=" * 60)
        if mode == 'mixed':
            print("Interactive Mode - Text + Voice Input")
            print("  • Type command and press ENTER (text input)")
            print("  • Type 'voice' then ENTER to switch to voice input")
            print("  • Type 'quit' to exit")
        elif mode == 'text':
            print("Interactive Mode - Text Input Only")
            print("  • Type command and press ENTER")
            print("  • Type 'quit' to exit")
        elif mode == 'voice':
            print("Interactive Mode - Voice Input Only")
            print("  • Speak your command (auto-detection)")
            print("  • Press Ctrl+C to exit")
        
        print("=" * 60)
        print("\nCommand Examples:")
        print("  • Add 1k resistor R1")
        print("  • Add 5V power source V1")
        print("  • Connect R1 and V1")
        print("  • Delete R1")
        print("=" * 60)
        
        current_mode = 'text'  # Current input mode
        
        while True:
            try:
                if mode == 'voice' or (mode == 'mixed' and current_mode == 'voice'):
                    # Voice input mode
                    if not mic_available:
                        print("\n[❌] Microphone not available, switching to text mode")
                        current_mode = 'text'
                        continue
                    
                    # Capture and recognize voice
                    recognized_text = capture_and_recognize_voice(ai_voice)
                    
                    if recognized_text:
                        # Execute the recognized command
                        print(f"[→] 执行指令: {recognized_text}")
                        result = ai_voice.execute_command(recognized_text)
                        
                        if result["success"]:
                            print(f"[✓ OK] {result['message']}")
                            if result.get('data'):
                                print(f"      Data: {result['data']}")
                        else:
                            print(f"[✗ FAIL] {result['message']}")
                            if result.get('needs_clarification'):
                                print(f"      {result.get('clarification_question', '')}")
                    else:
                        # Recognition failed
                        if mode == 'mixed':
                            print("[!] 语音识别失败，切换到文本模式 (输入 'voice' 可再次尝试语音)")
                            current_mode = 'text'
                        else:
                            print("[!] 语音识别失败，请重试")
                    
                    continue
                
                else:
                    # Text input mode
                    if mode == 'mixed':
                        prompt = f"\n[{current_mode.upper()}] > "
                    else:
                        prompt = "\n> "
                    
                    user_input = input(prompt).strip()
                    
                    # Handle mode switching
                    if mode == 'mixed' and user_input.lower() == 'voice':
                        if mic_available:
                            current_mode = 'voice'
                            print("[🎤] 已切换到语音模式 (语音识别后将自动返回文本模式)")
                            continue
                        else:
                            print("[❌] Microphone not available")
                            continue
                    
                    # Handle exit
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("Exiting...")
                        break
                    
                    if not user_input:
                        continue
                    
                    # Execute command
                    result = ai_voice.execute_command(user_input)
                    
                    if result["success"]:
                        print(f"[✓ OK] {result['message']}")
                        if result.get('data'):
                            print(f"      Data: {result['data']}")
                    else:
                        print(f"[✗ FAIL] {result['message']}")
                        if result.get('needs_clarification'):
                            print(f"      {result.get('clarification_question', '')}")
                    
            except KeyboardInterrupt:
                if mode == 'voice' or current_mode == 'voice':
                    print("\n[!] Interrupted. Switching to text mode...")
                    if mode == 'mixed':
                        current_mode = 'text'
                        continue
                    else:
                        break
                else:
                    print("\nProgram interrupted")
                    break
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def test_only():
    """Run tests only"""
    print("=" * 60)
    print("AI Voice Layer - Test Mode")
    print("=" * 60)
    
    try:
        from scr.data import get_data_layer
        from scr.ai_voice import AIVoiceLayer
        
        data_layer = get_data_layer()
        ai_voice = AIVoiceLayer(data_layer)
        
        # Test parsing
        print("\n[Test 1] Voice command parsing:")
        ai_voice.test_voice_system()
        
        # Test execution
        print("\n[Test 2] Command execution (data layer only):")
        data_layer.clear_all_data()
        
        test_commands = [
            "Add 1k resistor R1",
            "Add 5V power source V1", 
            "Connect R1 and V1",
            "Delete R1",
        ]
        
        for cmd in test_commands:
            result = ai_voice.execute_command(cmd)
            status = "OK" if result['success'] else "FAIL"
            print(f"  [{status}] {cmd} -> {result['message']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == '--test':
            test_only()
        elif arg == '--text':
            sys.exit(main('text'))
        elif arg == '--voice':
            sys.exit(main('voice'))
        elif arg in ['--help', '-h']:
            print(__doc__)
            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python run_voice.py [--test | --text | --voice | --help]")
            sys.exit(1)
    else:
        # Default: mixed mode (text + voice)
        sys.exit(main('mixed'))
