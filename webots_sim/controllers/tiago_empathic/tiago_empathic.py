"""
TIAGo Empathic Navigator - Uses ToM-based continuous path planning.
Empathic yielding emerges naturally from EFE computation with different alpha values.
"""

from controller import Supervisor
import math

# Import the ToM planner
import tom_planner
from tom_planner import ToMPlanner


class TiagoEmpathicNavigator:
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.name = self.robot.getName()

        print(f"{self.name}: Starting initialization...")

        # Get self node for position queries
        self.self_node = self.robot.getSelf()
        if not self.self_node:
            print(f"{self.name}: ERROR - Could not get self node!")

        # Parse custom data for goal, alpha, agent_id
        self._parse_custom_data()

        # Motor parameters
        self.max_speed = 5.0  # rad/s
        self.wheel_radius = 0.0985  # meters
        self.wheel_base = 0.4044    # meters

        # Control gains
        self.Kp_heading = 2.0
        self.max_angular_speed = 2.0
        self.max_linear_speed = 0.5  # m/s

        # State
        self.goal_reached = False
        self.has_fallen = False
        self.goal_tolerance = 0.3

        # Other robot tracking
        self.other_node = None
        self.other_name = None
        self.other_alpha = 0.5  # Default, will be updated

        # Initialize devices and find other robot
        self._init_devices()
        self._find_other_robot()

        # Auto-discover arena geometry and configure planner
        self._configure_planner_from_world()

        # Create ToM planner with continuous positions
        self.planner = ToMPlanner(
            agent_id=self.agent_id,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
            alpha=self.alpha
        )

        # Let simulation stabilize
        for _ in range(10):
            self.robot.step(self.timestep)

        print(f"{self.name}: Initialized (agent_id={self.agent_id}, alpha={self.alpha}, goal=({self.goal_x}, {self.goal_y}))")

    def _parse_custom_data(self):
        """Parse goal, alpha, agent_id from customData."""
        custom_data = self.robot.getCustomData()
        # Defaults
        self.goal_x = 1.25
        self.goal_y = 0.0
        self.alpha = 0.5
        self.agent_id = 0

        if custom_data:
            try:
                parts = custom_data.strip().split(',')
                if len(parts) >= 2:
                    self.goal_x = float(parts[0])
                    self.goal_y = float(parts[1])
                if len(parts) >= 3:
                    self.alpha = float(parts[2])
                if len(parts) >= 4:
                    self.agent_id = int(parts[3])
                print(f"{self.name}: Parsed customData - goal=({self.goal_x},{self.goal_y}), alpha={self.alpha}, agent_id={self.agent_id}")
            except ValueError as e:
                print(f"{self.name}: Parse error: {e}")

    def _init_devices(self):
        """Initialize motors and sensors."""
        # Wheel motors
        self.left_motor = self.robot.getDevice('wheel_left_joint')
        self.right_motor = self.robot.getDevice('wheel_right_joint')

        if not self.left_motor or not self.right_motor:
            print(f"{self.name}: ERROR - Wheel motors not found!")
            return

        # Set to velocity control mode
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Tuck arm close to body so it doesn't stick out
        arm_tucked = {
            'arm_1_joint': 0.07,
            'arm_2_joint': 1.02,
            'arm_3_joint': -3.16,
            'arm_4_joint': 2.02,
            'arm_5_joint': 1.32,
            'arm_6_joint': 0.0,
            'arm_7_joint': 1.41,
        }
        for joint_name, pos in arm_tucked.items():
            joint = self.robot.getDevice(joint_name)
            if joint:
                joint.setPosition(pos)

        print(f"{self.name}: Motors initialized, arm tucked")

    def _find_other_robot(self):
        """Find the other TIAGo robot and extract its alpha."""
        root = self.robot.getRoot()
        children = root.getField('children')

        for i in range(children.getCount()):
            node = children.getMFNode(i)
            if node.getTypeName() == 'Tiago':
                name_field = node.getField('name')
                if name_field:
                    name = name_field.getSFString()
                    if name != self.name:
                        self.other_node = node
                        self.other_name = name
                        # Try to extract other robot's alpha from its customData
                        custom_data_field = node.getField('customData')
                        if custom_data_field:
                            other_custom = custom_data_field.getSFString()
                            parts = other_custom.strip().split(',')
                            if len(parts) >= 3:
                                self.other_alpha = float(parts[2])
                        print(f"{self.name}: Found other robot: {name} (alpha={self.other_alpha})")
                        return

        print(f"{self.name}: No other TIAGo robot found (single robot mode)")

    def _discover_hazards(self):
        """Find all HazardObstacle nodes and extract positions/sizes."""
        hazards = []
        root = self.robot.getRoot()
        children = root.getField('children')
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            try:
                type_name = node.getTypeName()
            except Exception:
                continue
            if type_name == 'HazardObstacle':
                try:
                    pos = node.getField('translation').getSFVec3f()
                    size = node.getField('size').getSFVec3f()
                    # Store as (x_center, y_center, x_half_size, y_half_size)
                    hazards.append((pos[0], pos[1], size[0] / 2.0, size[1] / 2.0))
                except Exception as e:
                    print(f"{self.name}: Warning - could not read hazard: {e}")
        return hazards

    def _get_arena_bounds(self):
        """Read arena floor size to determine coordinate bounds."""
        root = self.robot.getRoot()
        children = root.getField('children')
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            try:
                type_name = node.getTypeName()
            except Exception:
                continue
            if type_name == 'RectangleArena':
                try:
                    floor_size = node.getField('floorSize').getSFVec2f()
                    hx = floor_size[0] / 2.0
                    hy = floor_size[1] / 2.0
                    return -hx, hx, -hy, hy
                except Exception as e:
                    print(f"{self.name}: Warning - could not read arena size: {e}")
        # Default: 5x2 arena
        return -2.5, 2.5, -1.0, 1.0

    def _configure_planner_from_world(self):
        """Auto-discover world geometry and configure the planner."""
        hazards = self._discover_hazards()
        x_min, x_max, y_min, y_max = self._get_arena_bounds()
        # Shrink bounds slightly so edge bins are inside the arena
        margin = 0.3
        tom_planner.configure(
            x_min + margin, x_max - margin,
            y_min, y_max,
            hazards=hazards
        )
        print(f"{self.name}: Discovered {len(hazards)} hazards, arena=[{x_min:.1f},{x_max:.1f}]x[{y_min:.1f},{y_max:.1f}]")

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def get_position(self):
        """Get current position from supervisor node."""
        if self.self_node:
            pos = self.self_node.getPosition()
            return pos[0], pos[1], pos[2]
        return 0, 0, 0

    def get_heading(self):
        """Get current heading from supervisor node rotation."""
        if self.self_node:
            rot = self.self_node.getOrientation()
            # Extract yaw from rotation matrix
            yaw = math.atan2(rot[3], rot[0])
            return yaw
        return 0

    def get_other_position(self):
        """Get other robot's position."""
        if self.other_node:
            pos = self.other_node.getPosition()
            return pos[0], pos[1]
        return None, None

    def get_other_goal(self):
        """Get other robot's goal from customData."""
        if self.other_node:
            custom_data_field = self.other_node.getField('customData')
            if custom_data_field:
                other_custom = custom_data_field.getSFString()
                parts = other_custom.strip().split(',')
                if len(parts) >= 2:
                    return float(parts[0]), float(parts[1])
        # Default: opposite goal
        return -self.goal_x, self.goal_y

    def navigate_to_target(self, current_x, current_y, current_heading, target_x, target_y, debug=False):
        """Navigate towards target position using differential drive.

        Key improvement: If target is behind the robot (heading error > 90deg),
        drive BACKWARD instead of turning around. This is much faster for yielding.
        """
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx*dx + dy*dy)

        # Check if target reached
        if distance < 0.15:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            if debug:
                print(f"  {self.name} NAV: At target (dist={distance:.2f})")
            return True

        # Compute desired heading (direction TO target)
        desired_heading = math.atan2(dy, dx)
        heading_error = self.normalize_angle(desired_heading - current_heading)

        # Check if target is behind us (heading error > 90 degrees)
        # If so, drive backward instead of turning around
        drive_backward = abs(heading_error) > math.pi / 2

        if drive_backward:
            # Flip the heading error to face away from target (we'll reverse)
            heading_error = self.normalize_angle(heading_error + math.pi)

        # Compute angular velocity (to align with target or away if reversing)
        angular = self.Kp_heading * heading_error
        angular = max(-self.max_angular_speed, min(self.max_angular_speed, angular))

        # Two-phase motor primitive: rotate in place first if heading
        # error is large, then translate once aligned. This prevents the
        # "spin + creep" behavior that makes lateral moves fail near
        # boundaries — the robot actually commits to the rotation before
        # moving, producing decisive lateral displacement.
        if abs(heading_error) > math.radians(15):
            linear = 0.0  # Pure rotation until aligned
        else:
            # Compute linear velocity (reduce when turning)
            alignment = max(0.2, math.cos(heading_error))
            linear = self.max_linear_speed * alignment

            # Slow down near target
            if distance < 0.5:
                linear *= distance / 0.5

        # If driving backward, negate linear velocity
        if drive_backward:
            linear = -linear

        # Convert to wheel velocities
        v_left = (linear - angular * self.wheel_base / 2.0) / self.wheel_radius
        v_right = (linear + angular * self.wheel_base / 2.0) / self.wheel_radius

        # Clamp and apply
        v_left = max(-self.max_speed, min(self.max_speed, v_left))
        v_right = max(-self.max_speed, min(self.max_speed, v_right))

        self.left_motor.setVelocity(v_left)
        self.right_motor.setVelocity(v_right)

        if debug:
            mode = "REVERSE" if drive_backward else "FORWARD"
            print(f"  {self.name} NAV [{mode}]: pos=({current_x:.2f},{current_y:.2f}) heading={math.degrees(current_heading):.0f}deg -> target=({target_x:.2f},{target_y:.2f}) dist={distance:.2f}")
            print(f"    heading_err={math.degrees(heading_error):.0f}deg linear={linear:.2f} angular={angular:.2f} v_left={v_left:.2f} v_right={v_right:.2f}")

        return False

    def run(self):
        """Main control loop."""
        print(f"{self.name}: Starting navigation to goal ({self.goal_x}, {self.goal_y})")

        plan_timer = 0
        plan_interval = 0.2  # Replan every 0.2 seconds (faster reaction)
        target_x, target_y = self.goal_x, self.goal_y
        last_print_time = 0
        last_action = ""

        while self.robot.step(self.timestep) != -1:
            current_time = self.robot.getTime()

            # Get current state
            x, y, z = self.get_position()
            heading = self.get_heading()

            # Check if fallen
            if z < -0.1:
                if not self.has_fallen:
                    print(f"{self.name}: FALLEN! z={z}")
                    self.has_fallen = True
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                continue

            # Check goal
            goal_dist = math.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)
            if goal_dist < self.goal_tolerance:
                if not self.goal_reached:
                    print(f"{'='*50}")
                    print(f"{self.name}: SUCCESS - Goal reached!")
                    print(f"{'='*50}")
                    self.goal_reached = True
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                continue

            # Get other robot state
            other_x, other_y = self.get_other_position()
            other_goal_x, other_goal_y = self.get_other_goal()

            # Replan periodically
            if current_time - plan_timer > plan_interval:
                plan_timer = current_time

                if other_x is not None:
                    # Use ToM planner with full state
                    target_x, target_y, action = self.planner.plan(
                        my_x=x,
                        my_y=y,
                        other_x=other_x,
                        other_y=other_y,
                        other_goal_x=other_goal_x,
                        other_goal_y=other_goal_y,
                        other_alpha=self.other_alpha
                    )
                    last_action = action
                else:
                    # No other robot - go straight to goal
                    target_x, target_y = self.goal_x, self.goal_y
                    last_action = "DIRECT"

            # Debug output every second
            if current_time - last_print_time > 1.0:
                other_str = f"({other_x:.2f},{other_y:.2f})" if other_x is not None else "N/A"
                # Show direction of planned move
                dx = target_x - x
                dy = target_y - y
                direction = ""
                if abs(dx) < 0.05 and abs(dy) < 0.05:
                    direction = "STAY"
                elif abs(dy) > abs(dx):
                    direction = "UP" if dy > 0 else "DOWN"
                else:
                    direction = "RIGHT" if dx > 0 else "LEFT"
                dist_between = math.sqrt((x - other_x)**2 + (y - other_y)**2) if other_x is not None else 0
                print(f"{self.name}: pos=({x:.2f},{y:.2f}) other={other_str} dist={dist_between:.2f} -> {direction} target=({target_x:.2f},{target_y:.2f}) | {last_action}")
                last_print_time = current_time

            # Navigate to target (debug every 2 seconds)
            debug_nav = (current_time - last_print_time < 0.1)  # Debug right after status print
            self.navigate_to_target(x, y, heading, target_x, target_y, debug=debug_nav)


if __name__ == "__main__":
    navigator = TiagoEmpathicNavigator()
    navigator.run()
