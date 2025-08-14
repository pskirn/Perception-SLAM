# Perception-SLAM

Quick Project Summary
What is this?
It's a system that figures out how a camera moved just by looking at the video - like GPS but using only images. The original thesis had a working version, but it had some problems we're fixing.
Main Problems We're Solving:

The depth estimation was noisy (couldn't handle plain walls well)
Features were jumping around between frames
The system didn't know when to use which algorithm
Too much noise in the 3D point clouds

What I've Done:

Designed 4 improvements to fix these problems
Created flowcharts showing exactly how each improvement works
Calculated that we should get 30% better accuracy
Written all the algorithms and explanations in the thesis

The Cool Part:
Instead of replacing everything, I made the existing system smarter. Like:

Now it knows when it's confused (confidence scores)
It remembers what it saw in the last frame (temporal tracking)
It adapts to different scenes (city streets vs nature)

Current Status:
✓ All the thinking and design work is done
✓ Math and algorithms are ready
✓ Expected results calculated
✗ No code written yet
✗ Not tested on real data
What Happens Next:
Someone needs to:

Take the original code
Add my improvements (following the flowcharts)
Test it on KITTI dataset
Show it works 30% better

Why This Matters:

Robots can navigate without GPS
Self-driving cars work better in tunnels
AR/VR apps track more accurately
Drones can map areas more precisely