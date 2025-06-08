import numpy as np
from stable_baselines3 import PPO
import threading
import time
import asyncio
from tdmclient import ClientAsync

class PPOThymioController:
    def __init__(self, model_path):
        # Load trained PPO model
        self.model = PPO.load(model_path, device="cpu")
        
        # Sensor data
        self.prox_sensor = None
        self.ground_sensor = None
        
        # Motor commands (will be set by PPO)
        self.left_motor = 0
        self.right_motor = 0
        
        # Control flags
        self.running = True
        self.sensors_ready = False
        
    def create_observation(self):
        """Convert sensor readings to observation format for PPO"""
        if self.prox_sensor is None or self.ground_sensor is None:
            return None
        
        # Create observation vector - adjust based on your training setup
        obs = np.array([
            self.prox_sensor[0],  # front-left
            self.prox_sensor[1],  # front-left-center  
            self.prox_sensor[2],  # front-center
            self.prox_sensor[3],  # front-right-center
            self.prox_sensor[4],  # front-right
            self.ground_sensor[0],  # left ground
            self.ground_sensor[1],  # right ground
        ], dtype=np.float32)
        
        # Normalize sensors (adjust based on your training normalization)
        obs = obs / 1000.0 #TODO verificar se é divir por 1000
        
        return obs
    
    def action_to_motors(self, action):
        """Convert PPO action to motor commands"""
        # Scale to Thymio motor range (-500 to 500)
        #TODO verificar se é para 500
        self.left_motor = int(np.clip(action[0] * 200, -500, 500))
        self.right_motor = int(np.clip(action[1] * 200, -500, 500))
    
    async def thymio_communication(self):
        """Handle Thymio sensor reading and motor control"""
        with ClientAsync() as client:
            async def comm_loop():
                with await client.lock() as node:
                    await node.wait_for_variables({"prox.horizontal", "prox.ground.reflected"})
                    print("Connected to Thymio - sensors ready")
                    self.sensors_ready = True
                    
                    while self.running:
                        # Read sensors
                        self.prox_sensor = list(node.v.prox.horizontal)
                        self.ground_sensor = list(node.v.prox.ground.reflected)
                        
                        # Apply motor commands from PPO
                        node.v.motor.left.target = self.left_motor
                        node.v.motor.right.target = self.right_motor
                        node.flush()
                        
                        await client.sleep(0.05)  # 20Hz update rate
            
            await comm_loop()
    
    def ppo_control_loop(self):
        """Main PPO control loop"""
        print("PPO controller starting...")
        
        # Wait for sensors to be ready
        while not self.sensors_ready:
            time.sleep(0.1)
        
        print("PPO taking control of Thymio!")
        step_count = 0
        
        try:
            while self.running:
                # Get current observation
                obs = self.create_observation()
                
                if obs is not None:
                    # PPO decides what to do
                    action, _states = self.model.predict(obs, deterministic=True)
                    
                    # Convert action to motor commands
                    self.action_to_motors(action)
                    
                    # Debug output
                    if step_count % 20 == 0:  # Print every second at 20Hz
                        print(f"Step {step_count}: "
                              f"Prox: {self.prox_sensor[:3]} | "
                              f"Ground: {self.ground_sensor} | "
                              f"Motors: L={self.left_motor}, R={self.right_motor}")
                    
                    step_count += 1
                
                time.sleep(0.05)  # 20Hz control frequency
                
        except KeyboardInterrupt:
            print("\nStopping PPO control...")
            self.stop()
    
    def stop(self):
        """Stop the robot safely"""
        self.running = False
        self.left_motor = 0
        self.right_motor = 0
        time.sleep(0.5)
    
    def run(self):
        """Start the PPO-controlled Thymio"""
        # Start Thymio communication in separate thread
        def run_async():
            asyncio.run(self.thymio_communication())
        
        thymio_thread = threading.Thread(target=run_async)
        thymio_thread.daemon = True
        thymio_thread.start()
        
        # Start PPO control loop
        self.ppo_control_loop()

def main():
    MODEL_PATH = "PPO_test_3.zip"
    
    print("Starting PPO-controlled Thymio...")
    print("Press Ctrl+C to stop")
    
    controller = PPOThymioController(MODEL_PATH)
    controller.run()

if __name__ == "__main__":
    main()