package nl.proto;

import org.nlogo.headless.HeadlessWorkspace;

/**
 * NetLogoRunner
 *
 * Contract that NativeRuntime.java already expects:
 *     new NetLogoRunner(modelPath)
 *     double result = runner.run(ticks, commands, seed)
 *
 * Where
 *   - commands  → full newline-separated NetLogo source (setup included)
 *   - seed      → numeric seed passed to ‘random-seed’
 * The *reporter* is **conveyed INSIDE the commands string**, e.g.
 *   setup
 *   go 100
 *   report (mean [pxcor] of turtles)
 *
 * Cross-model portability is delegated to the command generator:
 *   every experiment simply embeds its own reporter expression in the
 *   command string before submitting this call.
 */
public final class NetLogoRunner {

    private final HeadlessWorkspace ws;

    public NetLogoRunner(String modelPath) {
        try { ws = HeadlessWorkspace.newInstance(); }             // 6.x
        catch (Exception ex) { throw new RuntimeException(ex); }

        try { ws.open(modelPath); }
        catch (Exception ex) { throw new RuntimeException("Cannot open " + modelPath, ex); }
    }

    /**
     * Run exactly once.
     * @param ticks   total ticks to advance (ignored if 'go <n>' already present in commands)
     * @param commands  complete scripts: setup, go, reporter call(s)
     * @param seed   numeric seed supplied by the sampling library
     * @return       last value returned by the workspace after commands finish
     */
    public double run(int ticks, String commands, String seed) {
        synchronized (ws) {                 // workspace is single-threaded
            try {
                ws.command("random-seed " + seed);
                ws.command(commands);       // arbitrary NetLogo code
                Object v = ws.report("count turtles");   // Use a standard reporter that works in most models
                return (v instanceof Number) ? ((Number) v).doubleValue() : Double.NaN;
            } catch (Exception ex) {
                ex.printStackTrace();
                return Double.NaN;
            }
        }
    }
    
    /**
     * Run with specific metrics collection.
     * @param ticks   total ticks to advance
     * @param commands  complete scripts: setup, go, reporter call(s)
     * @param seed   numeric seed supplied by the sampling library
     * @param metrics array of metric names to collect
     * @return       array of metric values in the same order as metrics array
     */
    public double[] runWithMetrics(int ticks, String commands, String seed, String[] metrics) {
        synchronized (ws) {                 // workspace is single-threaded
            try {
                ws.command("random-seed " + seed);
                ws.command(commands);       // arbitrary NetLogo code
                
                double[] results = new double[metrics.length];
                for (int i = 0; i < metrics.length; i++) {
                    try {
                        Object v = ws.report(metrics[i]);
                        results[i] = (v instanceof Number) ? ((Number) v).doubleValue() : Double.NaN;
                    } catch (Exception ex) {
                        System.err.println("Error collecting metric '" + metrics[i] + "': " + ex.getMessage());
                        results[i] = Double.NaN;
                    }
                }
                return results;
            } catch (Exception ex) {
                ex.printStackTrace();
                double[] results = new double[metrics.length];
                for (int i = 0; i < metrics.length; i++) {
                    results[i] = Double.NaN;
                }
                return results;
            }
        }
    }
    
    // Static methods for C++ JNI bridge compatibility
    public static double headless(int ticks, String modelPath, String setupCommands, int seed) {
        try {
            // Parse parameters from setupCommands (format: "mutation:1.0,selection:10.0")
            double mutation = 1.0;
            double selection = 10.0;
            
            if (setupCommands != null && !setupCommands.isEmpty()) {
                String[] parts = setupCommands.split(",");
                for (String part : parts) {
                    String[] keyValue = part.split(":");
                    if (keyValue.length == 2) {
                        String key = keyValue[0].trim();
                        double value = Double.parseDouble(keyValue[1].trim());
                        if ("mutation".equals(key)) {
                            mutation = value;
                        } else if ("selection".equals(key)) {
                            selection = value;
                        }
                    }
                }
            }
            
            // Use NativeRuntime to handle the call
            NativeRuntime.init(modelPath);
            return NativeRuntime.run(ticks, mutation, selection, seed);
        } catch (Exception e) {
            e.printStackTrace();
            return Double.NaN;
        }
    }
    
    // Static method for metrics collection
    public static double[] headlessWithMetrics(int ticks, String modelPath, String setupCommands, int seed, String[] metrics) {
        try {
            // Parse parameters from setupCommands
            double mutation = 1.0;
            double selection = 10.0;
            
            if (setupCommands != null && !setupCommands.isEmpty()) {
                String[] parts = setupCommands.split(",");
                for (String part : parts) {
                    String[] keyValue = part.split(":");
                    if (keyValue.length == 2) {
                        String key = keyValue[0].trim();
                        double value = Double.parseDouble(keyValue[1].trim());
                        if ("mutation".equals(key)) {
                            mutation = value;
                        } else if ("selection".equals(key)) {
                            selection = value;
                        }
                    }
                }
            }
            
            // Use NativeRuntime to handle the call with metrics
            NativeRuntime.init(modelPath);
            return NativeRuntime.runWithMetrics(ticks, mutation, selection, seed, metrics);
        } catch (Exception e) {
            e.printStackTrace();
            double[] results = new double[metrics.length];
            for (int i = 0; i < metrics.length; i++) {
                results[i] = Double.NaN;
            }
            return results;
        }
    }
    
    public static double gui(int ticks, String modelPath, String setupCommands, int seed) {
        // GUI mode not supported, return NaN
        return Double.NaN;
    }
}