#!/bin/bash

echo "=== NVIDIA Nsight Compute Permission Setup for Linux ==="
echo ""
echo "This script configures your system to allow Nsight Compute (ncu) to access"
echo "GPU performance counters without requiring sudo."
echo ""
echo "Requirements: Linux display driver 418.43 or later"
echo ""
echo "Choose one of the following methods:"
echo ""
echo "1. PERMANENT - Via NVIDIA module (Recommended)"
echo "   Creates /etc/modprobe.d/nvidia-profiling.conf"
echo "   Requires reboot to take effect"
echo ""
echo "2. TEMPORARY - Reload kernel module (Advanced)"
echo "   Stops display manager, reloads NVIDIA modules"
echo "   Takes effect immediately but doesn't survive reboot"
echo ""
echo "3. TEMPORARY - Via perf_event_paranoid (Simple)"
echo "   Quick fix that works in most cases"
echo "   Takes effect immediately but doesn't survive reboot"
echo ""
echo "Press 1, 2, or 3 to apply a method, or any other key to exit:"

read -n 1 choice
echo ""

case $choice in
    1)
        echo "Applying PERMANENT method (NVIDIA module configuration)..."
        echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf

        echo ""
        echo "Rebuilding initrd for changes to take effect..."

        # Detect distribution and rebuild initrd accordingly
        if [ -f /etc/debian_version ]; then
            echo "Detected Debian-based system, using update-initramfs..."
            sudo update-initramfs -u -k all
        elif [ -f /etc/redhat-release ] || [ -f /etc/fedora-release ]; then
            echo "Detected RedHat-based system, using dracut..."
            sudo dracut --regenerate-all -f
        else
            echo "Could not detect distribution. You may need to manually rebuild initrd."
            echo "For Debian/Ubuntu: sudo update-initramfs -u -k all"
            echo "For RedHat/Fedora: sudo dracut --regenerate-all -f"
        fi

        echo ""
        echo "Done! Configuration saved. Please REBOOT for changes to take effect."
        echo "After reboot, Nsight Compute will work without sudo for all users."
        ;;

    2)
        echo "Applying TEMPORARY method (Reload kernel modules)..."
        echo "WARNING: This will temporarily stop your display manager!"
        echo "Press 'y' to continue or any other key to cancel:"

        read -n 1 confirm
        echo ""

        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo "Stopping display manager..."
            sudo systemctl isolate multi-user

            echo "Unloading NVIDIA modules..."
            sudo modprobe -rf nvidia_uvm nvidia_drm nvidia_modeset nvidia-vgpu-vfio nvidia 2>/dev/null

            # Try alternative module names for Ubuntu
            if [ $? -ne 0 ]; then
                echo "Trying Ubuntu-specific module names..."
                # Get the driver version
                NVIDIA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)
                if [ ! -z "$NVIDIA_VERSION" ]; then
                    sudo modprobe -rf nvidia_uvm nvidia_drm nvidia_modeset nvidia-${NVIDIA_VERSION} 2>/dev/null
                fi
            fi

            echo "Loading NVIDIA module with profiling enabled..."
            sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0

            echo "Restarting display manager..."
            sudo systemctl isolate graphical

            echo "Done! Nsight Compute should now work without sudo (until next reboot)."
        else
            echo "Cancelled."
        fi
        ;;

    3)
        echo "Applying TEMPORARY method (perf_event_paranoid)..."
        sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
        echo "Done! Nsight Compute should now work without sudo (until next reboot)."
        echo ""
        echo "Note: This method may not work on all systems. If you still see"
        echo "ERR_NVGPUCTRPERM, try method 1 (permanent) or method 2 (module reload)."
        ;;

    *)
        echo "No changes made. Exiting."
        exit 0
        ;;
esac

echo ""
echo "To verify the setting:"
echo "  - Check module parameter: cat /proc/driver/nvidia/params | grep RestrictProfilingToAdminUsers"
echo "    (should show 0 for unrestricted access)"
echo "  - Check perf setting: cat /proc/sys/kernel/perf_event_paranoid"
echo "    (should show 1 or lower)"
echo ""
echo "You can now run profiling without sudo!"
echo "Examples:"
echo "  - Direct: ncu --set full -o report ./your_cuda_program"
echo "  - CMake with Ninja: cmake --build build --target <target>_profile"
echo "  - Ninja directly: ninja -C build <target>_profile"
echo ""
echo "If you still encounter issues:"
echo "  - Make sure no processes are using NVIDIA modules (check with: lsof | grep nvidia)"
echo "  - Try method 1 (permanent) which is most reliable"
echo "  - Check NVIDIA driver version: nvidia-smi --query-gpu=driver_version --format=csv"