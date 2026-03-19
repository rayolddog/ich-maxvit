const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.createModule(.{
        .root_source_file = b.path("src/hu_tensor.zig"),
        .target           = target,
        .optimize         = optimize,
    });

    const lib = b.addLibrary(.{
        .name        = "hu_tensor",
        .linkage     = .dynamic,
        .root_module = mod,
    });

    b.installArtifact(lib);
}
