from llmpidtuner.models import FirstOrderPlant, PIDParams, SimulationSettings
from llmpidtuner.simulation import FirstOrderDelaySimulator


def main() -> None:
    plant = FirstOrderPlant(k=0.39, t=210)
    pid = PIDParams(kp=1.0, ki=0.1, kd=0.01)
    settings = SimulationSettings(setpoint=1, sim_time=2000, num_points=20000, time_delay=20)

    result = FirstOrderDelaySimulator(plant, pid, settings).run()
    print(f"IAE: {result.iae:.2f}")
    print(f"Final output: {result.output[-1]:.5f}")


if __name__ == "__main__":
    main()
