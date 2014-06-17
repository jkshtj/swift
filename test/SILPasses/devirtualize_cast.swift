// RUN: %swift %s -O3 -emit-sil
// Make sure we are not crashing on this one.

class X { func ping() {} }
class Y : X { override func ping() {} }

func foo(y : Y) {
  var x : X = y
  x.ping()
}

