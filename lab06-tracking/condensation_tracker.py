import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib import patches

from color_histogram import color_histogram
from propagate import propagate
from observe import observe
from resample import resample
from estimate import estimate


top_left = []
bottom_right = []

def line_select_callback(clk, rls):
    print(clk.xdata, clk.ydata)
    global top_left
    global bottom_right
    top_left = (int(clk.xdata), int(clk.ydata))
    bottom_right = (int(rls.xdata), int(rls.ydata))

def onkeypress(event):
    global top_left
    global bottom_right
    global img
    if event.key == 'q':
        print('final bbox', top_left, bottom_right)
        plt.close()

def toggle_selector(event):
    toggle_selector.RS.set_active(True)


def condensation_tracker(video_name, params):
    '''
    video_name - video name
    params - parameters
        - draw_plats        {0, 1} draw output plots throughout
        - hist_bin          1-255 number of histogram bins for each color: proper values 4,8,16
        - alpha             number in [0,1]; color histogram update parameter (0 = no update)
        - sigma_position    std. dev. of system model position noise w_i
        - sigma_observe     std. dev. of observation model noise v_i
        - num_particles     number of particles
        - model             {0,1} system model (0 = no motion, 1 = constant velocity)
        if using model = 1 then the following parameters are used:
            - sigma_velocity   std. dev. of system model velocity noise
            - initial_velocity initial velocity to set particles to
    '''
    # Choose video
    if video_name == "video1.avi":
        first_frame = 10
        last_frame = 42
    elif video_name == "video2.avi":
        first_frame = 3
        last_frame = 40
    elif video_name == "video3.avi":
        first_frame = 1
        last_frame = 60

    # Change this to where your data is
    data_dir = './ex6_data/'
    video_path = os.path.join(data_dir, video_name)
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(1, first_frame)
    ret, first_image = vidcap.read()
    fig, ax = plt.subplots(1)
    image = first_image
    frame_height = first_image.shape[0]
    frame_width = first_image.shape[1]

    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    ax.imshow(first_image)
    print(first_image.shape) # [height-120, width-160, ch]

    # select the tracking object by manually draw a bounding box
    toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )
    bbox = plt.connect('key_press_event', toggle_selector)
    key = plt.connect('key_press_event', onkeypress)
    plt.title("Draw a box then press 'q' to continue")
    plt.show()

    # calculate bbox size
    bbox_width = bottom_right[0] - top_left[0]
    bbox_height = bottom_right[1] - top_left[1]

    # Get initial color histogram
    hist = color_histogram(
        top_left[1], top_left[0],
        bottom_right[1], bottom_right[0],
        first_image, params["hist_bin"]
        )
    # ===========================================

    state_length = 2
    if(params["model"] == 1):
        state_length = 4

    # a prior mean state, a posterior mean state
    mean_state_a_prior = np.zeros([last_frame - first_frame + 1, state_length])
    mean_state_a_posterior = np.zeros([last_frame - first_frame + 1, state_length])
    # bounding box centre for x1, x2
    mean_state_a_prior[0, 0:2] = [(top_left[1]+bottom_right[1])/2., (top_left[0]+bottom_right[0])/2.]
    # use initial velocity for x3, x4 if model = 1
    if params["model"] == 1:
        mean_state_a_prior[0, 2:4] = params["initial_velocity"]
    print("Initial State: ")
    print(mean_state_a_prior[0,:])

    # Initialize Particles - for current frame
    particles = np.tile(mean_state_a_prior[0], (params["num_particles"], 1)) # [num_p, state_dim]
    # Init weigth of each particle - uniform weight
    particles_weight = np.ones([params["num_particles"], 1]) * 1./params["num_particles"] # [num_p, 1]

    fig, ax = plt.subplots(1)
    im = ax.imshow(first_image)
    plt.ion()


    # loop the video
    for i in range(last_frame - first_frame + 1):
        t = i + first_frame

        # ===== Propagate particles. =====
        # using system transition function and uncertainty 
        # s_t = A * S_t-1' + w_t-1
        particles = propagate(particles, frame_height, frame_width, params)

        # ===== Estimate the prior state of frame i. ===== 
        mean_state_a_prior[i, :] = estimate(particles, particles_weight)
        # print("Prior: ")
        # print(mean_state_a_prior[i, :])

        # Get new frame
        ret, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ===== Display the prior result =====
        if params["draw_plots"]:
            ax.set_title("Frame: %d" % t)
            im.set_data(frame)
            to_remove = []

            # Plot a prior particles - center of bbox of all particles
            new_plot = ax.scatter(particles[:, 1], particles[:, 0], color='blue', s=2)
            to_remove.append(new_plot)

            # Plot a prior estimation - center of bbox of mean prior of the last 10 frames
            for j in range(i-1, -1, -1):
                lwidth = 30 - 3 * (i-j)
                if lwidth > 0:
                    new_plot = ax.scatter(
                        mean_state_a_prior[j+1, 1],
                        mean_state_a_prior[j+1, 0],
                        color='blue', s=lwidth
                    )
                    to_remove.append(new_plot)
                # plot arrow - shift of mean prior
                if j != i:
                    new_plot = ax.plot(
                        [mean_state_a_prior[j, 1], mean_state_a_prior[j+1, 1]], 
                        [mean_state_a_prior[j, 0], mean_state_a_prior[j+1, 0]],
                        color='blue')
                    to_remove.append(new_plot[0])

            # Plot a prior bounding box
            if not np.any(np.isnan(mean_state_a_prior[i, :])):
                patch = ax.add_patch(
                    patches.Rectangle((
                        mean_state_a_prior[i, 1] - 0.5 * bbox_width,
                        mean_state_a_prior[i, 0] - 0.5 * bbox_height),
                    bbox_width, bbox_height,
                    fill=False, edgecolor='blue', lw=2
                    ))
                to_remove.append(patch)

        # ========== Observe ==========
        # for each s_t computer the chi2 dist of hist between itself and the target
        # and weight each s_t using gaussian distribution according to chi2-dist
        particles_weight = observe(
            particles, frame, bbox_height, bbox_width,
            params["hist_bin"], hist, params["sigma_observe"]
        )

        # Update estimation weighted sum of s_t
        mean_state_a_posterior[i, :] = estimate(particles, particles_weight)
        # print("Posterior: ")
        # print(mean_state_a_posterior[i, :])

        # ===== Update histogram color model =====
        # get mean_state_Hist
        hist_post = color_histogram(
            min(max(0, round(mean_state_a_posterior[i, 0]-0.5*bbox_height)), frame_height-1),
            min(max(0, round(mean_state_a_posterior[i, 1]-0.5*bbox_width)), frame_width-1),
            min(max(0, round(mean_state_a_posterior[i, 0]+0.5*bbox_height)), frame_height-1),
            min(max(0, round(mean_state_a_posterior[i, 1]+0.5*bbox_width)), frame_width-1),
            frame, params["hist_bin"]
            )
        # new_Hist = (1-alpha) * target_Hist + alpha * mean_state_Hist   
        hist = (1 - params["alpha"]) * hist + params["alpha"] * hist_post
        
        # Display the posterior result
        if params["draw_plots"]:
            # Plot updated estimation
            for j in range(i-1, -1, -1):
                lwidth = 30 - 3 * (i-j)
                if lwidth > 0:
                    new_plot = ax.scatter(
                        mean_state_a_posterior[j+1, 1],
                        mean_state_a_posterior[j+1, 0],
                        color='red', s=lwidth
                    )
                    to_remove.append(new_plot)
                if j != i:
                    new_plot = ax.plot(
                        [mean_state_a_posterior[j, 1], mean_state_a_posterior[j+1, 1]], 
                        [mean_state_a_posterior[j, 0], mean_state_a_posterior[j+1, 0]],
                        color='red')
                    to_remove.append(new_plot[0])
            
            # Plot updated bounding box
            if not np.any(np.isnan(mean_state_a_posterior[i, :])):
                patch = ax.add_patch(
                    patches.Rectangle((
                        mean_state_a_posterior[i, 1] - 0.5 * bbox_width, 
                        mean_state_a_posterior[i, 0] - 0.5 * bbox_height),
                    bbox_width, bbox_height,
                    fill=False, edgecolor='red', lw=2
                    ))
                to_remove.append(patch)


        # ===== RESAMPLE PARTICLES =====
        particles, particles_weight = resample(particles, particles_weight)

        if params["draw_plots"] and t != last_frame:
            plt.pause(0.2)
            # Remove previous element from plot
            for e in to_remove:
                e.remove()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    video_name = 'video2.avi'
    params = {
        "draw_plots": 1,
        "hist_bin": 16,
        "alpha": 0.8,
        "sigma_observe": 0.3,
        "model": 1,
        "num_particles": 40,            # 30, 40, 50
        "sigma_position": 15,           # 15, 10, 18
        "sigma_velocity": 4,            # 1, 1, 10
        "initial_velocity": (0, 15)    # (-8, -2) (2, 5) (0, 15)
    }
    # CONDitional DENSity propagaTION over time
    condensation_tracker(video_name, params)
