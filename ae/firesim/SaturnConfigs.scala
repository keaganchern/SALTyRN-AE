package firechip.chip

import org.chipsalliance.cde.config.{Config}
import saturn.common.{VectorParams}

class GENShuttleConfig(vlen: Int = 256, dlen: Int = 128, ncores: Int = 2) extends Config (
  new WithDefaultFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new saturn.shuttle.WithShuttleVectorUnit(vlen, dlen, VectorParams.genParams) ++
  new chipyard.config.WithSystemBusWidth(dlen) ++
  new shuttle.common.WithShuttleTileBeatBytes(dlen/8) ++
  new shuttle.common.WithNShuttleCores(ncores) ++
  new chipyard.config.AbstractConfig)
class GENV128ShuttleConfig extends Config(
  new GENShuttleConfig(vlen=128, dlen=64, ncores=1)
)
class GENV256DualShuttleConfig extends Config(
  new GENShuttleConfig(vlen=256, dlen=128, ncores=2)
)
class GENV512ShuttleConfig extends Config(
  new GENShuttleConfig(vlen=512, dlen=256, ncores=1)
)
