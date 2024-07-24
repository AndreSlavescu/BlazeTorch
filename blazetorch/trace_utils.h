#pragma once

#define CHECKEQ(cond) if (!(cond)) { return c10::nullopt; }
