/*
 * NativeRuntime.java
 * MIT 2024, Ultra-perf NetLogo driver
 *
 * Provides the minimal static gateway invoked directly
 * from pure JNI (C++) to avoid reflection cost.
 */

package nl.proto;

public final class NativeRuntime {
    private static NetLogoRunner runner = null;

    /** Called exactly once per process by JNI bridge. */
    public static void init(String modelPath) throws Exception {
        runner = new NetLogoRunner(modelPath);
    }

    /**
     * Core JNI fast entry:
     * Converts C++ parameters to NetLogo commands and returns result.
     */
    public static double run(int ticks, double paramA, double paramB, int seed) throws Exception {
        if (runner == null) {
            throw new IllegalStateException("NetLogoRunner not initialized");
        }
        
        // Build NetLogo command string from parameters
        String commands = String.format(
            "set mutation %f\n" +
            "set selection %f\n" +
            "setup\n" +
            "repeat %d [ go ]",
            paramA, paramB, ticks
        );
        
        return runner.run(ticks, commands, String.valueOf(seed));
    }
}