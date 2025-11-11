package com.judgmentlabs.judgeval;

public final class Version {

    private Version() {
    }

    /**
     * Gets the current SDK version.
     *
     * @return the version string
     */
    public static String getVersion() {
        Package pkg = Version.class.getPackage();
        return (pkg != null && pkg.getImplementationVersion() != null) ? pkg.getImplementationVersion() : "unknown";
    }
}
