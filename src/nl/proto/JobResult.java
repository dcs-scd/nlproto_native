package nl.proto;

public final class JobResult {
    public final JobMeta meta;
    public final double mean;
    public final double stdev;
    public final double entropy;

    public JobResult(JobMeta meta, double mean, double stdev, double entropy) {
        this.meta   = meta;
        this.mean   = mean;
        this.stdev  = stdev;
        this.entropy = entropy;
    }

    @Override
    public String toString() {
        return meta.jobId + "," + mean + "," + stdev + "," + entropy;
    }
}