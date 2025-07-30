package nl.proto;

public final class JobMeta {
    public final long jobId;
    public final int ticks;

    public JobMeta(long jobId, int ticks) {
        this.jobId = jobId;
        this.ticks = ticks;
    }
}