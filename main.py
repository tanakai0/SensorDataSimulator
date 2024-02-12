import time
from pathlib import Path

# self-made
import src.activity_model as activity_model
import src.anomaly_model as anomaly_model
import src.floor_plan as floor_plan
import src.sensor_model as sensor_model
import src.utils as utils
import src.constants as constants

working_path = Path().resolve()
# layout_data_path has the layouts
layout_data_path = working_path / constants.LAYOUT_DATA_FOLDER
# path has the layout data
path = Path(layout_data_path / "test_layout")
# data_save_path saves the data
data_save_path = utils.generate_data_folder(path, utils.present_date_and_time())

start_days = 0
end_days = 7
# end_days = 9 * 360

temp_time = time.time()


# floor plan (FP)
FP = floor_plan.FloorPlan()
FP.load_layout(path)
utils.pickle_dump(data_save_path, constants.FLOOR_PLAN, FP)


# models for Anomaly labels (AL)
anomaly_info = utils.generate_six_anomalies(path, data_save_path, end_days, show=False)
fall_s_place_bed = ["Bed"]
fall_w_parameters = {"num": anomaly_info["fall_w_num"], "mean_lie_down_seconds": 30}
fall_s_parameters = {
    "num": anomaly_info["fall_s_num"],
    "mean_lie_down_seconds": 30,
    "place": fall_s_place_bed,
}
AL_model = {
    "housebound_go_out_num": 1 / 14,
    "housebound_go_out_duration": 20,
    "housebound_use_the_phone_num": 1 / 3,
    "housebound_use_the_phone_duration": 10,
    "semi_bedridden_nap_duration": 40,
    "semi_bedridden_rest_duration": 60,
    "semi_bedridden_go_out_num": 1 / 14,
    "semi_bedridden_go_out_duration": 20,
    "wandering_num": anomaly_info["wandering_num"],
    "wandering_minute": anomaly_info["wandering_mean_minutes"],
    "fall_w_parameters": fall_w_parameters,
    "fall_s_parameters": fall_s_parameters,
    "forgetting_num": anomaly_info["forgetting_num"],
}
utils.pickle_dump(data_save_path, constants.ANOMALY_MODEL, AL_model)

# Activity sequence (AS)
state_anomaly_labels = {
    anomaly_model.BEING_HOUSEBOUND: anomaly_info["housebound_labels"],
    anomaly_model.BEING_SEMI_BEDRIDDEN: anomaly_info["semi_bedridden_labels"],
}
AS_model = activity_model.basic_activity_model
utils.pickle_dump(data_save_path, constants.ACTIVITY_MODEL, AS_model)
AS = utils.generate_activity_sequence(
    start_days,
    end_days,
    path,
    original_act_model=AS_model,
    state_anomaly_labels=state_anomaly_labels,
    anomaly_parameters=AL_model,
)
utils.save_activity_sequence(data_save_path, AS, file_name="AS")
utils.pickle_dump(data_save_path, constants.ACTIVITY_SEQUENCE, AS)
print("An activity sequence was generated. {} [s]".format(time.time() - temp_time))


# Walking trajectories (WT)
WT_model = activity_model.indoor_movement  # parameters about walking
utils.pickle_dump(data_save_path, constants.WALKING_MODEL, WT_model)
WT = utils.generate_walking_trajectories(
    path,
    AS,
    WT_model.stride,
    WT_model.step_speed,
    WT_model.prefer_foot,
    fall_w_parameters=AL_model["fall_w_parameters"],
    fall_s_parameters=AL_model["fall_s_parameters"],
)
utils.pickle_dump(data_save_path, constants.WALKING_TRAJECTORY, WT)
utils.save_walking_trajectoires(data_save_path, WT, constants.WALKING_TRAJECTORY)
print("Walking trajectories were generated. {} [s]".format(time.time() - temp_time))


# sensor data (SD)
sensors = sensor_model.test_sensors2  # for test_layout
utils.pickle_dump(data_save_path, constants.SENSOR_MODEL, sensors)
utils.save_layout(data_save_path, path, sensors=sensors, show=False, with_name_furniture_place=False)

motion_SD = utils.generate_motion_sensor_data(
    FP,
    sensors,
    AS,
    WT,
    sampling_seconds=0.1,
    sync_reference_point=AS[0].start,
    body_radius=10,
)
print("Motion sensor data was simulated. {}[s]".format(time.time() - temp_time))

# decide which activities are forgot and how long the forgetting continues
forgetting_labels = anomaly_model.forgetting_labels(AS, WT, AL_model["forgetting_num"])
cost_SD = utils.generate_cost_sensor_data(
    sensors,
    AS,
    WT,
    sampling_seconds=1,
    sync_reference_point=AS[0].start,
    forgetting_labels=forgetting_labels,
)
print("Cost sensor data was simulated. {}[s]".format(time.time() - temp_time))

sorted_sensor_data = sorted(motion_SD + cost_SD, key=lambda x: x[0])
print("Sort sensor data.")
utils.save_binary_sensor_data(
    data_save_path, sensors, sorted_sensor_data, filename=constants.SENSOR_DATA
)
utils.pickle_dump(data_save_path, constants.SENSOR_DATA, sorted_sensor_data)
print("All sensor data was simulated. {}[s]".format(time.time() - temp_time))

# Anomaly labels (AL)
AL = {}
AL[anomaly_model.MMSE] = anomaly_info["MMSE"]
AL[anomaly_model.BEING_HOUSEBOUND] = anomaly_info["housebound_labels"]
AL[anomaly_model.BEING_SEMI_BEDRIDDEN] = anomaly_info["semi_bedridden_labels"]
AL[anomaly_model.FORGETTING] = forgetting_labels
wandering_labels = []
fall_w_labels = []
fall_s_labels = []
for i, wt in enumerate(WT):
    if wt.walking_type == constants.WANDERING_WALKING:
        wandering_labels.append((i, wt.start_time, wt.end_time))
    if wt.fall_w:
        fall_w_labels.append(((wt.timestamp[wt.fall_w_index]), wt.lie_down_seconds_w))
    if wt.fall_s:
        fall_s_labels.append(((wt.timestamp[wt.fall_s_index]), wt.lie_down_seconds_s))
AL[anomaly_model.WANDERING] = wandering_labels
AL[anomaly_model.FALL_WHILE_WALKING] = fall_w_labels
AL[anomaly_model.FALL_WHILE_STANDING] = fall_s_labels
utils.pickle_dump(data_save_path, constants.ANOMALY, AL)

print("Finished. {}[s]".format(time.time() - temp_time))
